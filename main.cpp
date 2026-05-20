#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickPaintedItem>
#include <QPainter>
#include <QAudioSource>
#include <QMediaDevices>
#include <QAudioDevice>
#include <QTimer>
#include <QImage>
#include <QThread>
#include <QMutex>
#include <QPermissions>
#include <vector>
#include <complex>
#include <cmath>
#include <limits>
#include <algorithm>

const int TARGET_SAMPLE_RATE = 96000;
const int FALLBACK_SAMPLE_RATE = 48000;
const int MAX_SPECTROGRAM_WIDTH = 1200;
const int FFT_SIZE = 2048;
const int HOP_SIZE = FFT_SIZE / 4;

class FFT {
public:
    static void forward(std::vector<std::complex<double>>& data) {
        int n = data.size();
        if (n <= 1 || (n & (n-1)) != 0) return; // power of 2 only

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

class SpectrogramItem : public QQuickPaintedItem {
    Q_OBJECT
public:
    explicit SpectrogramItem(QQuickItem* parent = nullptr) : QQuickPaintedItem(parent) {
        setRenderTarget(QQuickPaintedItem::FramebufferObject);
        img = QImage(MAX_SPECTROGRAM_WIDTH, 512, QImage::Format_RGB32);
        clear();
    }

    Q_INVOKABLE void addColumn(const std::vector<double>& magnitudes) {
        QMutexLocker locker(&mutex);
        if (magnitudes.size() != static_cast<size_t>(img.height())) {
            img = QImage(MAX_SPECTROGRAM_WIDTH, magnitudes.size(), QImage::Format_RGB32);
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
            QColor color = QColor::fromHsv(240 - (intensity * 200 / 255), 255, std::min(255, intensity + 40));
            img.setPixelColor(colIndex, img.height() - 1 - y, color);
        }
        colIndex++;
        update();
    }

    Q_INVOKABLE void clear() {
        QMutexLocker locker(&mutex);
        clear_internal();
        update();
    }

    void paint(QPainter* painter) override {
        QMutexLocker locker(&mutex);
        painter->drawImage(boundingRect(), img);
    }

private:
    void clear_internal() {
        img.fill(Qt::black);
        colIndex = 0;
    }

    QImage img;
    int colIndex = 0;
    QMutex mutex;
};

class CaviaAnalyzer : public QObject {
    Q_OBJECT
    Q_PROPERTY(QStringList translations READ translations NOTIFY translationsChanged)
    Q_PROPERTY(bool isRecording READ isRecording NOTIFY isRecordingChanged)
    Q_PROPERTY(int currentSpecies READ currentSpecies WRITE setCurrentSpecies NOTIFY currentSpeciesChanged)
    Q_PROPERTY(QString status READ status NOTIFY statusChanged)

public:
    enum Species { GuineaPig = 0, Capybara = 1 };
    Q_ENUM(Species)

    explicit CaviaAnalyzer(QObject* parent = nullptr);
    ~CaviaAnalyzer();

    void setSpectrogram(SpectrogramItem* item) { spectrogram = item; }

    QStringList translations() const { return m_translations; }
    bool isRecording() const { return m_isRecording; }
    int currentSpecies() const { return m_species; }
    QString status() const { return m_status; }

    void setCurrentSpecies(int idx);
    Q_INVOKABLE void toggleRecording();

signals:
    void translationsChanged();
    void isRecordingChanged();
    void currentSpeciesChanged();
    void statusChanged();

private:
    void startRecording();
    void stopRecording();
    void processAudioChunk(const QByteArray& chunk);
    void analyzeFrame(const int16_t* data, int frameStart, int sampleRate);
    void classifyCall(double freq, double duration, double timestamp);

    QAudioSource* audioSource = nullptr;
    QIODevice* audioDevice = nullptr;
    SpectrogramItem* spectrogram = nullptr;

    QByteArray audioBuffer;
    QStringList m_translations;
    QString m_status = "Ready";
    bool m_isRecording = false;
    Species m_species = GuineaPig;
    int sampleRate = TARGET_SAMPLE_RATE;

    QTimer* stopTimer = nullptr;
};

CaviaAnalyzer::CaviaAnalyzer(QObject* parent) : QObject(parent) {
    stopTimer = new QTimer(this);
    stopTimer->setSingleShot(true);
    connect(stopTimer, &QTimer::timeout, this, &CaviaAnalyzer::stopRecording);
}

CaviaAnalyzer::~CaviaAnalyzer() {
    if (m_isRecording) stopRecording();
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

    // ----- REQUEST RUNTIME MIC PERMISSIONS -----
    QMicrophonePermission micPermission;
    auto permissionStatus = qApp->checkPermission(micPermission);

    if (permissionStatus == Qt::PermissionStatus::Undetermined) {
        m_status = "Requesting microphone access...";
        emit statusChanged();
        qApp->requestPermission(micPermission, this, &CaviaAnalyzer::startRecording);
        return;
    }

    if (permissionStatus == Qt::PermissionStatus::Denied) {
        m_status = "Microphone permission denied! Cannot record.";
        emit statusChanged();
        return;
    }


    m_translations.clear();
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

    if (sampleRate < TARGET_SAMPLE_RATE) {
        m_status = QString("Recording @ %1 kHz (Warning: USVs >24kHz blocked)").arg(sampleRate / 1000.0);
    } else {
        m_status = QString("Recording @ %1 kHz (High-Fidelity USV Enabled)").arg(sampleRate / 1000.0);
    }

    emit isRecordingChanged();
    emit statusChanged();

    connect(audioDevice, &QIODevice::readyRead, this, [this]() {
        processAudioChunk(audioDevice->readAll());
    });

    stopTimer->start(15000);
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

    if (audioBuffer.size() > FFT_SIZE * 20 * sizeof(int16_t)) {
        audioBuffer.remove(0, sampleOffset * sizeof(int16_t) - FFT_SIZE * sizeof(int16_t));
        sampleOffset = FFT_SIZE;
    }
}

void CaviaAnalyzer::analyzeFrame(const int16_t* data, int offset, int sr) {
    std::vector<std::complex<double>> frame(FFT_SIZE);
    for (int i = 0; i < FFT_SIZE; ++i) {
        double window = 0.5 * (1 - std::cos(2 * M_PI * i / (FFT_SIZE - 1)));
        frame[i] = std::complex<double>(data[offset + i] * window, 0.0);
    }

    FFT::forward(frame);

    std::vector<double> mags(FFT_SIZE / 2);
    double maxMag = 1e-8;
    int peakBin = 0;
    double peakFreq = 0;
    double minFreq = (m_species == Capybara) ? 50.0 : 200.0;
    double maxFreq = 40000.0;

    for (int k = 0; k < FFT_SIZE / 2; ++k) {
        double freq = k * static_cast<double>(sr) / FFT_SIZE;
        if (freq < minFreq || freq > maxFreq) {
            mags[k] = 0;
            continue;
        }
        double mag = std::abs(frame[k]);
        mags[k] = mag;
        if (mag > maxMag) {
            maxMag = mag;
            peakBin = k;
            peakFreq = freq;
        }
    }

    for (double& m : mags) m = std::min(1.0, m / (maxMag * 1.5));
    if (spectrogram) spectrogram->addColumn(mags);

    static bool inCall = false;
    static double callStart = 0;
    static double freqSum = 0;
    static int callFrames = 0;

    double currentTime = offset / static_cast<double>(sr);
    bool strongSignal = maxMag > 25000;

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
        if (duration >= 0.025) {
            classifyCall(avgFreq, duration, callStart);
        }
    }
}

void CaviaAnalyzer::classifyCall(double freq, double duration, double timestamp) {
    QString callType;
    if (m_species == GuineaPig) {
        if (freq > 22000) callType = "Pup Distress USV";
        else if (freq >= 3500) callType = "Chirrup / Scream (Alarm/Distress)";
        else if (freq > 500) callType = "Whistle / Wheek (Food Anticipation)";
        else if (freq > 400) callType = "Chutter (Exploration)";
        else if (freq >= 270) callType = "Purr / Durr (Contentment)";
        else return;
    } else {
        if (freq > 25000) {
            callType = "Ultrasonic Emission (Distress)";
        } else {
            struct Centroid { QString name; double freq; double dur; };
            std::vector<Centroid> centroids = {
                {"Whistle (Isolation)", 2868, 0.10},
                {"Bark (Alarm)", 1746, 0.15},
                {"Cry (Contact)", 1467, 0.33},
                {"Squeal (Agonistic)", 2037, 0.48},
                {"Whine (Conflict)", 1944, 1.15}
            };
            double bestDist = std::numeric_limits<double>::max();
            for (const auto& c : centroids) {
                double d = std::hypot((freq - c.freq), (duration - c.dur) * 1000);
                if (d < bestDist) {
                    bestDist = d;
                    callType = c.name;
                }
            }
        }
    }

    QString entry = QString::asprintf("[%.2fs] %s | %.0f Hz | %.2fs", timestamp, qPrintable(callType), freq, duration);
    if (m_translations.isEmpty() || !m_translations.last().contains(QString::asprintf("[%.2fs]", timestamp))) {
        m_translations.prepend(entry);
        emit translationsChanged();
    }
}

const char* qmlData = R"QML(
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Bioacoustics 1.0

ApplicationWindow {
    visible: true
    width: 480
    height: 800
    title: "Neuroethological Caviomorph Engine"
    color: "#0d1117"

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 12

        RowLayout {
            Text { text: "Species Profile:"; color: "white"; font.bold: true }
            ComboBox {
                Layout.fillWidth: true
                model: ["Domestic Guinea Pig (C. porcellus)", "Capybara (H. hydrochaeris)"]
                currentIndex: backend.currentSpecies
                onCurrentIndexChanged: backend.currentSpecies = currentIndex
                enabled: !backend.isRecording
            }
        }

        Button {
            Layout.fillWidth: true
            height: 56
            text: backend.isRecording ? "STOP RECORDING" : "START RECORDING"
            font.bold: true
            font.pixelSize: 16
            onClicked: backend.toggleRecording()

            background: Rectangle {
                color: backend.isRecording ? "#c42b1c" : "#238636"
                radius: 6
            }
            contentItem: Text {
                text: parent.text
                color: "white"
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                font: parent.font
            }
        }

        Text {
            text: backend.status
            color: "#58a6ff"
            font.pixelSize: 13
            Layout.alignment: Qt.AlignHCenter
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 300
            color: "black"
            border.color: "#30363d"

            Spectrogram {
                id: spectro
                anchors.fill: parent
                objectName: "spectroItem"
            }
        }

        Text {
            text: "Detected Live Calls"
            color: "white"
            font.bold: true
            font.pixelSize: 15
        }

        ListView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            model: backend.translations
            clip: true
            spacing: 4
            delegate: Text {
                text: modelData
                color: "#39ff6e"
                font.pixelSize: 14
                wrapMode: Text.Wrap
            }
        }
        

        Text {
            text: "<a href='https://github.com/ewanp2025/CaviaBioacoustics'>Privacy Policy</a>"
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
