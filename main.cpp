#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickPaintedItem>
#include <QPainter>
#include <QAudioSource>
#include <QMediaDevices>
#include <QAudioDevice>
#include <QAudioSink>
#include <QBuffer>
#include <QTimer>
#include <QImage>
#include <QMutex>
#include <QPermissions>
#include <QDir>
#include <QStandardPaths>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QDateTime>
#include <QFile>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <random>

const int TARGET_SAMPLE_RATE = 48000;
const int FALLBACK_SAMPLE_RATE = 44100;
const int MAX_SPECTROGRAM_WIDTH = 1200;
const int FFT_SIZE = 1024;
const int HOP_SIZE = FFT_SIZE / 4;


class MLClassifier : public QObject {
    Q_OBJECT
public:
    explicit MLClassifier(QObject* parent = nullptr) : QObject(parent) {}

    Q_INVOKABLE void loadModel(const QString& modelPath) {
        qDebug() << "[ML] Model would load from:" << modelPath;
        m_modelLoaded = true;
    }

    QString predict(double avgFreq, double duration, double maxMag, const QString& species) {
        if (!m_modelLoaded) return {}; 
        
        
        qDebug() << "ML Inference triggered with Freq:" << avgFreq << "Dur:" << duration << "Mag:" << maxMag;
        return "ML: High-confidence " + species + " vocalization";
    }

private:
    bool m_modelLoaded = false;
};


class SpectrogramItem : public QQuickPaintedItem {
    Q_OBJECT
public:
    explicit SpectrogramItem(QQuickItem* parent = nullptr);
    Q_INVOKABLE void addColumn(const std::vector<double>& magnitudes);
    Q_INVOKABLE void clear();
    void paint(QPainter* painter) override;

private:
    void clear_internal();
    QImage img;
    int colIndex = 0;
    QMutex mutex;
};

SpectrogramItem::SpectrogramItem(QQuickItem* parent) : QQuickPaintedItem(parent) {
    setRenderTarget(QQuickPaintedItem::FramebufferObject);
    img = QImage(MAX_SPECTROGRAM_WIDTH, 512, QImage::Format_RGB32);
    clear();
}

void SpectrogramItem::addColumn(const std::vector<double>& magnitudes) {
    QMutexLocker locker(&mutex);
    if (magnitudes.size() != static_cast<size_t>(img.height())) {
        img = QImage(MAX_SPECTROGRAM_WIDTH, static_cast<int>(magnitudes.size()), QImage::Format_RGB32);
        clear_internal();
    }

    if (colIndex >= img.width()) {
        QImage newImg = img.copy(1, 0, img.width()-1, img.height());
        newImg.fill(Qt::black);
        img = newImg;
        colIndex = img.width() - 1;
    }

    for (size_t y = 0; y < magnitudes.size() && y < static_cast<size_t>(img.height()); ++y) {
        int intensity = std::clamp(static_cast<int>(magnitudes[y] * 255.0), 0, 255);
        int hue = 280 - (intensity * 240 / 255); 
        QColor color = QColor::fromHsv(hue, 255, std::min(255, intensity + 50));
        img.setPixelColor(colIndex, img.height() - 1 - y, color);
    }
    colIndex++;
    update();
}

void SpectrogramItem::clear() {
    QMutexLocker locker(&mutex);
    clear_internal();
    update();
}

void SpectrogramItem::clear_internal() {
    img.fill(Qt::black);
    colIndex = 0;
}

void SpectrogramItem::paint(QPainter* painter) {
    QMutexLocker locker(&mutex);
    painter->drawImage(boundingRect(), img);
}


class FFT {
public:
    static void forward(std::vector<std::complex<double>>& data) {
        int n = data.size();
        if (n <= 1 || (n & (n-1)) != 0) return;
        bitReverse(data);
        for (int len = 2; len <= n; len <<= 1) {
            double ang = 2 * M_PI / len;
            std::complex<double> wlen(std::cos(ang), std::sin(-ang));
            for (int i = 0; i < n; i += len) {
                std::complex<double> w(1, 0);
                for (int j = 0; j < len/2; ++j) {
                    auto u = data[i + j];
                    auto v = data[i + j + len/2] * w;
                    data[i + j] = u + v;
                    data[i + j + len/2] = u - v;
                    w *= wlen;
                }
            }
        }
    }
private:
    static void bitReverse(std::vector<std::complex<double>>& data) {
        int n = data.size();
        for (int i = 1, j = 0; i < n; ++i) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(data[i], data[j]);
        }
    }
};


class CaviaAnalyzer : public QObject {
    Q_OBJECT
    Q_PROPERTY(QStringList translations READ translations NOTIFY translationsChanged)
    Q_PROPERTY(bool isRecording READ isRecording NOTIFY isRecordingChanged)
    Q_PROPERTY(int currentSpecies READ currentSpecies WRITE setCurrentSpecies NOTIFY currentSpeciesChanged)
    Q_PROPERTY(QString status READ status NOTIFY statusChanged)
    Q_PROPERTY(QStringList savedSessions READ savedSessions NOTIFY savedSessionsChanged)

public:
    enum Species { GuineaPig = 0, Capybara = 1 };
    Q_ENUM(Species)

    explicit CaviaAnalyzer(QObject* parent = nullptr);
    ~CaviaAnalyzer();

    void setSpectrogram(SpectrogramItem* item) { spectrogram = item; }

    QStringList translations() const { return m_translations; }
    QStringList savedSessions() const { return m_savedSessions; }
    bool isRecording() const { return m_isRecording; }
    int currentSpecies() const { return m_species; }
    QString status() const { return m_status; }

    Q_INVOKABLE void setCurrentSpecies(int idx);
    Q_INVOKABLE void toggleRecording();
    Q_INVOKABLE void playCall(const QString& callType);
    Q_INVOKABLE void saveCurrentSession();
    Q_INVOKABLE void clearCurrentSession();

signals:
    void translationsChanged();
    void isRecordingChanged();
    void currentSpeciesChanged();
    void statusChanged();
    void savedSessionsChanged();

private:
    void startRecording();
    void stopRecording();
    void processAudioChunk(const QByteArray& chunk);
    void analyzeFrame(const int16_t* data, int offset, int sr);
    void classifyCall(double freq, double duration, double timestamp, double maxMag);
    void saveSessionToDisk();

    MLClassifier* mlClassifier = nullptr;
    SpectrogramItem* spectrogram = nullptr;

    QAudioSource* audioSource = nullptr;
    QIODevice* audioDevice = nullptr;
    QAudioSink* m_synthSink = nullptr;
    QBuffer* m_synthBuffer = nullptr;
    QByteArray m_synthData;

    QByteArray audioBuffer;
    QStringList m_translations;
    QStringList m_savedSessions;
    QString m_status = "Ready";
    bool m_isRecording = false;
    bool m_isPlaying = false;
    Species m_species = GuineaPig;
    int sampleRate = TARGET_SAMPLE_RATE;

    QJsonArray currentSessionData;
};

CaviaAnalyzer::CaviaAnalyzer(QObject* parent) : QObject(parent) {
    mlClassifier = new MLClassifier(this);
}

CaviaAnalyzer::~CaviaAnalyzer() {
    if (m_isRecording) stopRecording();
    if (m_synthSink) m_synthSink->deleteLater();
    if (m_synthBuffer) m_synthBuffer->deleteLater();
}

void CaviaAnalyzer::setCurrentSpecies(int idx) {
    Species s = static_cast<Species>(idx);
    if (m_species != s) {
        m_species = s;
        emit currentSpeciesChanged();
    }
}

void CaviaAnalyzer::toggleRecording() {
    if (m_isRecording) stopRecording();
    else startRecording();
}

void CaviaAnalyzer::startRecording() {
    if (m_isRecording) return;

    QMicrophonePermission micPermission;
    auto permissionStatus = qApp->checkPermission(micPermission);

    if (permissionStatus == Qt::PermissionStatus::Undetermined) {
        m_status = "Requesting microphone access...";
        emit statusChanged();
        qApp->requestPermission(micPermission, this, &CaviaAnalyzer::startRecording);
        return;
    }

    if (permissionStatus == Qt::PermissionStatus::Denied) {
        m_status = "Microphone permission denied!";
        emit statusChanged();
        return;
    }

    m_translations.clear();
    currentSessionData = QJsonArray();
    if (spectrogram) spectrogram->clear();
    audioBuffer.clear();
    emit translationsChanged();

    QAudioFormat format;
    format.setChannelCount(1);
    format.setSampleFormat(QAudioFormat::Int16);

    auto devices = QMediaDevices::audioInputs();
    if (devices.isEmpty()) {
        m_status = "No microphone found!";
        emit statusChanged();
        return;
    }

    QAudioDevice device = devices.first();
    format.setSampleRate(TARGET_SAMPLE_RATE);

    if (!device.isFormatSupported(format)) {
        format.setSampleRate(FALLBACK_SAMPLE_RATE);
        sampleRate = FALLBACK_SAMPLE_RATE;
        if (!device.isFormatSupported(format)) {
            m_status = "Unsupported audio format";
            emit statusChanged();
            return;
        }
    } else {
        sampleRate = TARGET_SAMPLE_RATE;
    }

    audioSource = new QAudioSource(device, format, this);
    audioDevice = audioSource->start();

    if (!audioDevice) {
        m_status = "Failed to start recording";
        emit statusChanged();
        return;
    }

    m_isRecording = true;
    m_status = QString("Recording @ %1 kHz").arg(sampleRate / 1000.0);
    emit isRecordingChanged();
    emit statusChanged();

    connect(audioDevice, &QIODevice::readyRead, this, [this]() {
        processAudioChunk(audioDevice->readAll());
    });
}

void CaviaAnalyzer::stopRecording() {
    if (!m_isRecording) return;
    if (audioSource) audioSource->stop();
    m_isRecording = false;
    m_status = "Ready";
    emit isRecordingChanged();
    emit statusChanged();

    if (audioDevice) {
        disconnect(audioDevice, nullptr, this, nullptr);
        audioDevice = nullptr;
    }
}

void CaviaAnalyzer::processAudioChunk(const QByteArray& chunk) {
    audioBuffer.append(chunk);
    const int16_t* data = reinterpret_cast<const int16_t*>(audioBuffer.constData());
    int totalSamples = audioBuffer.size() / sizeof(int16_t);
    static int sampleOffset = 0;

    while (sampleOffset + FFT_SIZE <= totalSamples) {
        analyzeFrame(data, sampleOffset, sampleRate);
        sampleOffset += HOP_SIZE;
    }

    if (audioBuffer.size() > FFT_SIZE * 30 * sizeof(int16_t)) {
        audioBuffer.remove(0, (sampleOffset - FFT_SIZE) * sizeof(int16_t));
        sampleOffset = FFT_SIZE;
    }
}

void CaviaAnalyzer::analyzeFrame(const int16_t* data, int offset, int sr) {
    if (m_isPlaying) return;

    std::vector<std::complex<double>> frame(FFT_SIZE);
    for (int i = 0; i < FFT_SIZE; ++i) {
        double window = 0.5 * (1 - std::cos(2 * M_PI * i / (FFT_SIZE - 1)));
        frame[i] = std::complex<double>(data[offset + i] * window, 0.0);
    }

    std::vector<std::complex<double>> fftData = frame;
    FFT::forward(fftData);

    std::vector<double> mags(FFT_SIZE / 2);
    double maxMag = 1e-8;
    double sumMag = 0;
    double peakFreq = 0;

    double minFreq = (m_species == Capybara) ? 50.0 : 200.0;

    for (int k = 0; k < FFT_SIZE / 2; ++k) {
        double freq = k * static_cast<double>(sr) / FFT_SIZE;
        if (freq < minFreq || freq > 24000) {
            mags[k] = 0;
            continue;
        }
        double mag = std::abs(fftData[k]);
        mags[k] = mag;
        sumMag += mag;
        if (mag > maxMag) {
            maxMag = mag;
            peakFreq = freq;
        }
    }

    double avgMag = sumMag / (FFT_SIZE / 2);
    for (double& m : mags) m = std::min(1.0, m / (maxMag * 1.5));

    if (spectrogram) spectrogram->addColumn(mags);

    static bool inCall = false;
    static double callStart = 0;
    static double freqSum = 0;
    static int callFrames = 0;

    bool strongSignal = (maxMag > 14000) && (maxMag > avgMag * 6.0);
    double currentTime = offset / static_cast<double>(sr);

    if (strongSignal) {
        if (!inCall) {
            inCall = true;
            callStart = currentTime;
            freqSum = 0;
            callFrames = 0;
        }
        freqSum += peakFreq;
        callFrames++;
    } else if (inCall) {
        inCall = false;
        double duration = (callFrames * HOP_SIZE) / static_cast<double>(sr);
        double avgFreq = freqSum / callFrames;
        classifyCall(avgFreq, duration, callStart, maxMag);
    }
}

void CaviaAnalyzer::classifyCall(double freq, double duration, double timestamp, double maxMag) {
    if (duration < 0.02) return;

    QString meaning;
    QString speciesStr = (m_species == GuineaPig) ? "Guinea Pig" : "Capybara";

  
    QString mlPrediction = mlClassifier->predict(freq, duration, maxMag, speciesStr);
    
    if (!mlPrediction.isEmpty()) {
        meaning = mlPrediction;
    } else {
        
        if (m_species == GuineaPig) {
            if (freq >= 4500 && duration < 0.05)      meaning = "Ambiguity / Low-Level Stress";
            else if (freq >= 1000 && freq <= 4000 && duration < 0.1) meaning = "Aerial Predator Alarm";
            else if (freq >= 800 && duration >= 0.5 && duration <= 0.7) meaning = "Extreme Fear / Acute Aggression";
            else if (freq >= 500 && duration >= 0.25 && duration <= 1.0) meaning = "Food Anticipation / Excitement";
            else if (freq >= 500 && duration >= 0.25 && duration <= 0.5) meaning = "Alert / Warning / Pain";
            else if (freq >= 400 && freq <= 500 && duration >= 0.05 && duration <= 0.1) meaning = "Exploration / Mild Frustration";
            else if (freq >= 261 && freq <= 476 && duration >= 0.05) meaning = "Contact Seeking / Freezing";
        } else { 
            if (duration > 1.0 && duration < 2.0 && freq > 2000) meaning = "Pup Isolation / Attraction";
            else if (duration > 0.08 && duration < 0.16) meaning = "Predator Alarm / Alert";
            else if (duration > 0.2 && duration < 0.5) meaning = "Distress / Isolation";
            else if (duration > 0.02 && duration < 0.08) meaning = "Spatial Monitoring / Group Cohesion";
        }
    }

    if (!meaning.isEmpty()) {
        QString entry = QString::asprintf("[%.2fs] %s", timestamp, qPrintable(meaning));
        m_translations.prepend(entry);
        emit translationsChanged();

        QJsonObject call;
        call["timestamp"] = timestamp;
        call["meaning"] = meaning;
        call["freq"] = freq;
        call["duration"] = duration;
        currentSessionData.append(call);
    }
}

void CaviaAnalyzer::saveCurrentSession() {
    if (currentSessionData.isEmpty()) return;
    saveSessionToDisk();
    m_savedSessions.prepend(QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm") + 
                            " — " + QString::number(currentSessionData.size()) + " calls");
    emit savedSessionsChanged();
}

void CaviaAnalyzer::saveSessionToDisk() {
    QString path = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation) + "/CaviaBioacoustics";
    QDir().mkpath(path);
    QString filename = path + "/session_" + QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + ".json";

    QJsonObject root;
    root["species"] = (m_species == GuineaPig) ? "Guinea Pig" : "Capybara";
    root["date"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    root["calls"] = currentSessionData;

    QFile file(filename);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
        qDebug() << "Session saved to:" << filename;
    }
}

void CaviaAnalyzer::clearCurrentSession() {
    m_translations.clear();
    currentSessionData = QJsonArray();
    emit translationsChanged();
}


void CaviaAnalyzer::playCall(const QString& callType) {
    if (m_synthSink) {
        m_synthSink->stop();
        m_synthSink->deleteLater();
        m_synthBuffer->deleteLater();
    }

    m_synthData.clear();
    int sr = TARGET_SAMPLE_RATE;
    double totalDurationSecs = 0.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    auto appendTone = [&](double startFreq, double endFreq, double durationSecs, bool hasHarmonics, double noiseLevel) {
        int samples = static_cast<int>(durationSecs * sr);
        double phase = 0.0;
        for (int i = 0; i < samples; ++i) {
            double t = static_cast<double>(i) / sr;
            double currentFreq = startFreq + (endFreq - startFreq) * (t / durationSecs);
            phase += 2 * M_PI * currentFreq / sr;

            double wave = std::sin(phase);
            if (hasHarmonics) {
                wave += 0.5 * std::sin(2 * phase);
                wave += 0.25 * std::sin(3 * phase);
            }
            if (noiseLevel > 0) wave += noiseLevel * dis(gen);

            double envelope = 1.0;
            if (i < 200) envelope = i / 200.0;
            if (i > samples - 200) envelope = (samples - i) / 200.0;

            int16_t sample = static_cast<int16_t>(8000 * wave * envelope);
            m_synthData.append(reinterpret_cast<const char*>(&sample), sizeof(int16_t));
        }
        totalDurationSecs += durationSecs;
    };

    auto appendSilence = [&](double durationSecs) {
        int samples = static_cast<int>(durationSecs * sr);
        int16_t zero = 0;
        for (int i = 0; i < samples; ++i) {
            m_synthData.append(reinterpret_cast<const char*>(&zero), sizeof(int16_t));
        }
        totalDurationSecs += durationSecs;
    };

    
    if (callType == "Purr") {
        for(int i=0; i<10; ++i) { appendTone(300, 300, 0.05, false, 0.1); appendSilence(0.016); }
    } else if (callType == "Chutter") {
        for(int i=0; i<4; ++i) { appendTone(450, 480, 0.075, true, 0.2); appendSilence(0.05); }
    } else if (callType == "Wheek") {
        appendTone(500, 3000, 0.5, true, 0.0);
    } else if (callType == "Squeal") {
        appendTone(500, 1000, 0.4, true, 0.3);
    } else if (callType == "Scream") {
        appendTone(800, 3000, 0.6, true, 0.8);
    } else if (callType == "Chirp") {
        for(int i=0; i<3; ++i) { appendTone(5000, 5000, 0.02, false, 0.0); appendSilence(0.05); }
    } else if (callType == "Chirrup") {
        appendTone(3500, 1000, 0.07, false, 0.0);
    }
    
    else if (callType == "Capy Click") {
        for(int i=0; i<6; ++i) { appendTone(200, 200, 0.01, false, 1.0); appendSilence(0.1); }
    } else if (callType == "Capy Bark") {
        appendTone(300, 8000, 0.13, true, 1.0);
    } else if (callType == "Capy Whistle") {
        appendTone(2800, 2800, 1.5, true, 0.0);
    } else if (callType == "Capy Whine") {
        appendTone(1900, 1500, 1.15, true, 0.3);
    } else if (callType == "Tooth-Chatter") {
        for(int i=0; i<15; ++i) { appendTone(100, 100, 0.02, false, 2.0); appendSilence(0.03); }
    }

    m_synthBuffer = new QBuffer(&m_synthData, this);
    m_synthBuffer->open(QIODevice::ReadOnly);

    QAudioFormat format;
    format.setSampleRate(sr);
    format.setChannelCount(1);
    format.setSampleFormat(QAudioFormat::Int16);

    m_synthSink = new QAudioSink(QMediaDevices::defaultAudioOutput(), format, this);
    m_synthSink->start(m_synthBuffer);

    m_isPlaying = true;
    QTimer::singleShot(static_cast<int>(totalDurationSecs * 1000) + 300, this, [this](){ m_isPlaying = false; });
}


const char* qmlData = R"QML(
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Bioacoustics 1.0

ApplicationWindow {
    visible: true
    width: 480
    height: 920
    title: "Cavia Bioacoustics"
    color: "#0d1117"

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 12

        RowLayout {
            Text { text: "Species:"; color: "white"; font.bold: true }
            ComboBox {
                Layout.fillWidth: true
                model: ["Guinea Pig", "Capybara"]
                currentIndex: backend.currentSpecies
                onCurrentIndexChanged: backend.currentSpecies = currentIndex
                enabled: !backend.isRecording
            }
        }

        Button {
            Layout.fillWidth: true
            height: 60
            text: backend.isRecording ? "STOP RECORDING" : "START RECORDING"
            font.bold: true; font.pixelSize: 18
            onClicked: backend.toggleRecording()
            background: Rectangle { color: backend.isRecording ? "#c42b1c" : "#238636"; radius: 8 }
            contentItem: Text { text: parent.text; color: "white"; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter; font: parent.font }
        }

        Text { text: backend.status; color: "#58a6ff"; font.pixelSize: 14; Layout.alignment: Qt.AlignHCenter }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 260
            color: "black"
            border.color: "#30363d"
            Spectrogram { id: spectro; anchors.fill: parent; objectName: "spectroItem" }
        }

        RowLayout {
            Button { text: "Save Session"; Layout.fillWidth: true; onClicked: backend.saveCurrentSession() }
            Button { text: "Clear"; Layout.fillWidth: true; onClicked: backend.clearCurrentSession() }
        }

        TabBar {
            id: bar
            Layout.fillWidth: true
            TabButton { text: "Live Translations" }
            TabButton { text: "Saved Sessions" }
        }

        StackLayout {
            currentIndex: bar.currentIndex
            Layout.fillWidth: true
            Layout.fillHeight: true

            ListView {
                clip: true
                model: backend.translations
                delegate: Text { text: modelData; color: "#39ff6e"; font.pixelSize: 14; wrapMode: Text.Wrap }
            }

            ListView {
                clip: true
                model: backend.savedSessions
                delegate: Text { text: modelData; color: "#8b949e"; font.pixelSize: 14 }
            }
        }

        Flow {
            Layout.fillWidth: true
            spacing: 8
            Repeater {
                model: backend.currentSpecies === 0 ? 
                    ["Purr", "Chutter", "Wheek", "Squeal", "Scream", "Chirp", "Chirrup"] : 
                    ["Capy Click", "Capy Bark", "Capy Whistle", "Capy Whine", "Tooth-Chatter"]
                Button {
                    text: modelData
                    onClicked: backend.playCall(modelData)
                    background: Rectangle { color: "#2f363d"; radius: 6; border.color: "#8b949e" }
                    contentItem: Text { text: parent.text; color: "white"; padding: 6 }
                }
            }
        }
        
        Text {
            text: "<a href='https://raw.githubusercontent.com/ewanp2025/CaviaBioacoustics/main/PRIVACY.md'>Privacy Policy</a>"
            color: "#58a6ff"
            font.pixelSize: 13
            Layout.alignment: Qt.AlignHCenter
            onLinkActivated: (link) => Qt.openUrlExternally(link)
        }
    }
}
)QML";

int main(int argc, char *argv[]) {
    QGuiApplication app(argc, argv);

    qmlRegisterType<SpectrogramItem>("Bioacoustics", 1, 0, "Spectrogram");

    QQmlApplicationEngine engine;
    CaviaAnalyzer backend;
    engine.rootContext()->setContextProperty("backend", &backend);
    engine.loadData(qmlData);

    if (engine.rootObjects().isEmpty()) return -1;

    QObject* root = engine.rootObjects().first();
    if (auto* spectro = root->findChild<SpectrogramItem*>("spectroItem")) {
        backend.setSpectrogram(spectro);
    }

    return app.exec();
}

#include "main.moc"
