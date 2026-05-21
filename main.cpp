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
#include <QThread>
#include <QMutex>
#include <QPermissions>
#include <vector>
#include <complex>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>

const int TARGET_SAMPLE_RATE = 48000;
const int FALLBACK_SAMPLE_RATE = 44100;
const int MAX_SPECTROGRAM_WIDTH = 1200;
const int FFT_SIZE = 1024; 
const int HOP_SIZE = FFT_SIZE / 4;

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
    Q_INVOKABLE void playCall(const QString& callType);

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
    QAudioSink* m_synthSink = nullptr;
    QBuffer* m_synthBuffer = nullptr;
    QByteArray m_synthData;

    SpectrogramItem* spectrogram = nullptr;

    QByteArray audioBuffer;
    QStringList m_translations;
    QString m_status = "Ready";
    bool m_isRecording = false;
    bool m_isPlaying = false; // Flag to prevent self-detection
    Species m_species = GuineaPig;
    int sampleRate = TARGET_SAMPLE_RATE;
};

CaviaAnalyzer::CaviaAnalyzer(QObject* parent) : QObject(parent) {
    // Removed the stopTimer entirely for continuous recording
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

void CaviaAnalyzer::playCall(const QString& callType) {
    if (m_synthSink) {
        m_synthSink->stop();
        m_synthSink->deleteLater();
        m_synthBuffer->deleteLater();
    }

    m_synthData.clear();
    int sr = TARGET_SAMPLE_RATE;
    double totalDurationSecs = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    // Advanced Synthesizer: Adds harmonics and noise for realism
    auto appendTone = [&](double startFreq, double endFreq, double durationSecs, bool hasHarmonics, double noiseLevel) {
        int samples = durationSecs * sr;
        double phase = 0;
        for(int i = 0; i < samples; ++i) {
            double t = (double)i / sr;
            double currentFreq = startFreq + (endFreq - startFreq) * (t / durationSecs);
            phase += 2 * M_PI * currentFreq / sr;
            
            double wave = std::sin(phase);
            if (hasHarmonics) {
                wave += 0.5 * std::sin(2 * phase); // 2nd harmonic
                wave += 0.25 * std::sin(3 * phase); // 3rd harmonic
            }
            if (noiseLevel > 0) {
                wave += noiseLevel * dis(gen); // Add chaotic noise
            }
            
            // Envelope: Fade in/out to prevent popping
            double envelope = 1.0;
            if (i < 200) envelope = i / 200.0;
            if (i > samples - 200) envelope = (samples - i) / 200.0;

            int16_t sample = 8000 * wave * envelope; // Halved amplitude to prevent clipping with harmonics
            m_synthData.append(reinterpret_cast<const char*>(&sample), sizeof(int16_t));
        }
        totalDurationSecs += durationSecs;
    };

    auto appendSilence = [&](double durationSecs) {
        int samples = durationSecs * sr;
        int16_t zero = 0;
        for(int i = 0; i < samples; ++i) {
            m_synthData.append(reinterpret_cast<const char*>(&zero), sizeof(int16_t));
        }
        totalDurationSecs += durationSecs;
    };

    // Construct the calls with appropriate harmonic profiles
    if (callType == "Purr") {
        for(int i=0; i<10; ++i) { appendTone(300, 300, 0.05, false, 0.1); appendSilence(0.016); }
    } else if (callType == "Chutter") {
        for(int i=0; i<4; ++i) { appendTone(450, 480, 0.075, true, 0.2); appendSilence(0.05); }
    } else if (callType == "Wheek") {
        appendTone(500, 3000, 0.5, true, 0.0); // Rich harmonics
    } else if (callType == "Squeal") {
        appendTone(500, 1000, 0.4, true, 0.3); // Harmonics + minor noise
    } else if (callType == "Scream") {
        appendTone(800, 3000, 0.6, true, 0.8); // High noise for chaos
    } else if (callType == "Chirp") {
        for(int i=0; i<3; ++i) { appendTone(5000, 5000, 0.02, false, 0.0); appendSilence(0.05); }
    } else if (callType == "Chirrup") {
        appendTone(3500, 1000, 0.07, false, 0.0);
    } 
    // CAPYBARA CALLS
    else if (callType == "Capy Click") {
        for(int i=0; i<6; ++i) { appendTone(200, 200, 0.01, false, 1.0); appendSilence(0.1); } // Broadband pulse
    } else if (callType == "Capy Bark") {
        appendTone(300, 8000, 0.13, true, 1.0); // Explosive sharp noise
    } else if (callType == "Capy Whistle") {
        appendTone(2800, 2800, 1.5, true, 0.0); // Pure 1.5s tone
    } else if (callType == "Capy Whine") {
        appendTone(1900, 1500, 1.15, true, 0.3); 
    } else if (callType == "Tooth-Chatter") {
        for(int i=0; i<15; ++i) { appendTone(100, 100, 0.02, false, 2.0); appendSilence(0.03); } // Pure mechanical clacks
    }

    m_synthBuffer = new QBuffer(&m_synthData, this);
    m_synthBuffer->open(QIODevice::ReadOnly);

    QAudioFormat format;
    format.setSampleRate(sr);
    format.setChannelCount(1);
    format.setSampleFormat(QAudioFormat::Int16);

    m_synthSink = new QAudioSink(QMediaDevices::defaultAudioOutput(), format, this);
    m_synthSink->start(m_synthBuffer);

    // Mute the analyzer while playing to prevent self-detection
    m_isPlaying = true;
    QTimer::singleShot((totalDurationSecs * 1000) + 200, this, [this](){ m_isPlaying = false; });
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
    if (m_isPlaying) return; // Ignore input while speaking to prevent feedback loops

    std::vector<std::complex<double>> frame(FFT_SIZE);
    for (int i = 0; i < FFT_SIZE; ++i) {
        double window = 0.5 * (1 - std::cos(2 * M_PI * i / (FFT_SIZE - 1)));
        frame[i] = std::complex<double>(data[offset + i] * window, 0.0);
    }

    FFT::forward(frame);

    std::vector<double> mags(FFT_SIZE / 2);
    double maxMag = 1e-8;
    double sumMag = 0;
    int peakBin = 0;
    double peakFreq = 0;
    double minFreq = (m_species == Capybara) ? 50.0 : 200.0;
    double maxFreq = 24000.0; 

    for (int k = 0; k < FFT_SIZE / 2; ++k) {
        double freq = k * static_cast<double>(sr) / FFT_SIZE;
        if (freq < minFreq || freq > maxFreq) {
            mags[k] = 0;
            continue;
        }
        double mag = std::abs(frame[k]);
        mags[k] = mag;
        sumMag += mag;
        if (mag > maxMag) {
            maxMag = mag;
            peakBin = k;
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

    double currentTime = offset / static_cast<double>(sr);
    
    // NOISE GATE: Volume must be high, AND it must be 6x louder than background noise
    bool strongSignal = (maxMag > 15000) && (maxMag > avgMag * 6.0);

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
        
        classifyCall(avgFreq, duration, callStart);
    }
}

void CaviaAnalyzer::classifyCall(double freq, double duration, double timestamp) {
    QString meaning;
    
    // Ignore micro-transients
    if (duration < 0.02) return; 

    if (m_species == GuineaPig) {
        // Output semantics rather than names
        if (freq >= 4500 && duration < 0.05) {
            meaning = "Ambiguity / Low-Level Stress";
        } else if (freq >= 1000 && freq <= 4000 && duration < 0.1) {
            meaning = "Aerial Predator Alarm";
        } else if (freq >= 800 && duration >= 0.5 && duration <= 0.7) {
            meaning = "Extreme Fear / Acute Aggression";
        } else if (freq >= 500 && duration >= 0.25 && duration <= 1.0) {
            meaning = "Food Anticipation / Excitement";
        } else if (freq >= 500 && duration >= 0.25 && duration <= 0.5) {
            meaning = "Alert / Warning / Pain";
        } else if (freq >= 400 && freq <= 500 && duration >= 0.05 && duration <= 0.1) {
            meaning = "Exploration / Mild Frustration";
        } else if (freq >= 261 && freq <= 476 && duration >= 0.05) {
            meaning = "Contact Seeking / Freezing";
        } else {
            return;
        }
    } else {
        // Precise duration filtering for Capybaras
        if (duration > 1.0 && duration < 2.0 && freq > 2000) {
            meaning = "Pup Isolation / Attraction"; // Whistle
        } else if (duration > 0.08 && duration < 0.16) {
            meaning = "Predator Alarm / Alert"; // Bark
        } else if (duration > 0.2 && duration < 0.5) {
            meaning = "Distress / Isolation"; // Cry/Squeal
        } else if (duration > 0.02 && duration < 0.08) {
            meaning = "Spatial Monitoring / Group Cohesion"; // Click
        } else {
            return; // Failed to match capybara profile
        }
    }

    QString entry = QString::asprintf("[%.2fs] %s", timestamp, qPrintable(meaning));
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
                contentItem: Text { text: parent.displayText; color: "white"; font: parent.font; verticalAlignment: Text.AlignVCenter; leftPadding: 12 }    
            }
        }

        Button {
            Layout.fillWidth: true
            height: 56
            text: backend.isRecording ? "STOP RECORDING" : "START RECORDING"
            font.bold: true; font.pixelSize: 16
            onClicked: backend.toggleRecording()
            background: Rectangle { color: backend.isRecording ? "#c42b1c" : "#238636"; radius: 6 }
            contentItem: Text { text: parent.text; color: "white"; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter; font: parent.font }
        }

        Text { text: backend.status; color: "#58a6ff"; font.pixelSize: 13; Layout.alignment: Qt.AlignHCenter }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 250
            color: "black"; border.color: "#30363d"
            Spectrogram { id: spectro; anchors.fill: parent; objectName: "spectroItem" }
        }

        Text { text: "Semantic Translations"; color: "white"; font.bold: true; font.pixelSize: 15; visible: backend.translations.length > 0 }

        ListView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            model: backend.translations
            clip: true; spacing: 4
            delegate: Text { text: modelData; color: "#39ff6e"; font.pixelSize: 14; wrapMode: Text.Wrap }
        }

        Rectangle {
            Layout.fillWidth: true; Layout.preferredHeight: 1; color: "#30363d" 
        }

        Text { text: "Vocal Synthesizer (Talk Back)"; color: "white"; font.bold: true; font.pixelSize: 15 }
        
        Flow {
            Layout.fillWidth: true
            spacing: 8
            
            // Dynamically change buttons based on selected species
            Repeater {
                model: backend.currentSpecies === 0 ? ["Purr", "Chutter", "Wheek", "Squeal", "Scream", "Chirp", "Chirrup"] : ["Capy Click", "Capy Whistle", "Capy Bark", "Capy Whine", "Tooth-Chatter"]
                Button {
                    text: modelData
                    onClicked: backend.playCall(modelData)
                    background: Rectangle { color: "#2f363d"; radius: 6; border.color: "#8b949e"; border.width: 1 }
                    contentItem: Text { text: parent.text; color: "white"; padding: 6 }
                }
            }
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
