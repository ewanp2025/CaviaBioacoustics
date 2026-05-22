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
#include <QDataStream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <random>

// Config Constants
const QList<int> PREFERRED_SAMPLE_RATES = {192000, 96000, 48000, 44100};
const int MAX_SPECTROGRAM_WIDTH = 1200;
const int FFT_SIZE = 1024; 
const int HOP_SIZE = FFT_SIZE / 8; 
const int WAVEFORM_SAMPLES = 512;

// Biomimetic ML Classifier 
class MLClassifier : public QObject {
    Q_OBJECT
public:
    explicit MLClassifier(QObject* parent = nullptr) : QObject(parent) {}
    Q_INVOKABLE void loadModel(const QString& modelPath) { m_modelLoaded = true; }
    QString predict(double avgFreq, double duration, double maxMag, const std::vector<double>& pulseRates, const QString& species) {
        if (!m_modelLoaded) return {}; 
        return "ML Prediction";
    }
private:
    bool m_modelLoaded = false;
};

// LIVE WAVEFORM VISUALIZER
class WaveformItem : public QQuickPaintedItem {
    Q_OBJECT
public:
    explicit WaveformItem(QQuickItem* parent = nullptr);
    Q_INVOKABLE void updateWaveform(const std::vector<double>& samples);
    void paint(QPainter* painter) override;

private:
    std::vector<double> waveformData;
    QMutex mutex;
};

WaveformItem::WaveformItem(QQuickItem* parent) : QQuickPaintedItem(parent) {
    setRenderTarget(QQuickPaintedItem::FramebufferObject);
}

void WaveformItem::updateWaveform(const std::vector<double>& samples) {
    QMutexLocker locker(&mutex);
    waveformData = samples;
    update();
}

void WaveformItem::paint(QPainter* painter) {
    QMutexLocker locker(&mutex);
    if (waveformData.empty()) return;

    painter->fillRect(boundingRect(), Qt::black);
    QPen pen(QColor("#58a6ff"), 2.0);
    painter->setPen(pen);

    double centerY = height() / 2.0;
    double scaleY = height() * 0.45;

    for (size_t i = 1; i < waveformData.size(); ++i) {
        double x1 = (i-1) * width() / (waveformData.size() - 1);
        double x2 = i * width() / (waveformData.size() - 1);
        double y1 = centerY - waveformData[i-1] * scaleY;
        double y2 = centerY - waveformData[i] * scaleY;
        painter->drawLine(QPointF(x1, y1), QPointF(x2, y2));
    }

    painter->setPen(QPen(QColor(40, 40, 40), 1));
    for (int i = 1; i < 4; ++i) {
        double y = i * height() / 4.0;
        painter->drawLine(0, y, width(), y);
    }
}

// LOGARITHMIC SPECTROGRAM VISUALIZER
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
        double db = 20 * std::log10(std::max(magnitudes[y], 1e-8));
        int intensity = std::clamp(static_cast<int>((db + 70) * 3.5), 0, 255);
        int hue = 280 - (intensity * 240 / 255); 
        QColor color = QColor::fromHsv(hue, 255, std::min(255, intensity + 50));
        img.setPixelColor(colIndex, img.height() - 1 - y, color);
    }
    colIndex++;
    update();
}

void SpectrogramItem::clear() { QMutexLocker locker(&mutex); clear_internal(); update(); }
void SpectrogramItem::clear_internal() { img.fill(Qt::black); colIndex = 0; }
void SpectrogramItem::paint(QPainter* painter) { QMutexLocker locker(&mutex); painter->drawImage(boundingRect(), img); }

// FFT ALGORITHM
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

// MASTER ANALYZER ENGINE
class CaviaAnalyzer : public QObject {
    Q_OBJECT
    Q_PROPERTY(QStringList translations READ translations NOTIFY translationsChanged)
    Q_PROPERTY(bool isRecording READ isRecording NOTIFY isRecordingChanged)
    Q_PROPERTY(int currentSpecies READ currentSpecies WRITE setCurrentSpecies NOTIFY currentSpeciesChanged)
    Q_PROPERTY(QString status READ status NOTIFY statusChanged)
    Q_PROPERTY(QStringList savedSessions READ savedSessions NOTIFY savedSessionsChanged)
    
    Q_PROPERTY(QString petName READ petName WRITE setPetName NOTIFY petNameChanged)
    Q_PROPERTY(double sensitivity READ sensitivity WRITE setSensitivity NOTIFY sensitivityChanged)
    Q_PROPERTY(bool saveRawWav READ saveRawWav WRITE setSaveRawWav NOTIFY saveRawWavChanged)

public:
    enum Species { GuineaPig = 0, Capybara = 1 };
    Q_ENUM(Species)

    explicit CaviaAnalyzer(QObject* parent = nullptr);
    ~CaviaAnalyzer();

    void setSpectrogram(SpectrogramItem* item) { spectrogram = item; }
    void setWaveform(WaveformItem* item) { waveform = item; }

    QStringList translations() const { return m_translations; }
    QStringList savedSessions() const { return m_savedSessions; }
    bool isRecording() const { return m_isRecording; }
    int currentSpecies() const { return m_species; }
    QString status() const { return m_status; }
    
    QString petName() const { return m_petName; }
    double sensitivity() const { return m_sensitivity; }
    bool saveRawWav() const { return m_saveRawWav; }

    Q_INVOKABLE void setCurrentSpecies(int idx);
    Q_INVOKABLE void setPetName(const QString& name);
    Q_INVOKABLE void setSensitivity(double val);
    Q_INVOKABLE void setSaveRawWav(bool val);
    
    Q_INVOKABLE void toggleRecording();
    Q_INVOKABLE void playCall(const QString& callType);
    Q_INVOKABLE void saveCurrentSession();
    Q_INVOKABLE void clearCurrentSession();
    Q_INVOKABLE void calibrateNoiseFloor();

signals:
    void translationsChanged();
    void isRecordingChanged();
    void currentSpeciesChanged();
    void statusChanged();
    void savedSessionsChanged();
    void petNameChanged();
    void sensitivityChanged();
    void saveRawWavChanged();

private:
    void startRecording();
    void stopRecording();
    void processAudioChunk(const QByteArray& chunk);
    void analyzeFrame(const int16_t* data, int offset, int sr);
    double computeAutocorrelationPulseRate(const std::vector<double>& waveformSamples, int sr);
    void classifyCall(double freq, double duration, double timestamp, double maxMag, double pulseRate);
    void writeWavFile(const QString& filename);

    MLClassifier* mlClassifier = nullptr;
    SpectrogramItem* spectrogram = nullptr;
    WaveformItem* waveform = nullptr;

    QAudioSource* audioSource = nullptr;
    QIODevice* audioDevice = nullptr;
    QAudioSink* m_synthSink = nullptr;
    QBuffer* m_synthBuffer = nullptr;
    QByteArray m_synthData;

    QByteArray audioBuffer;
    QByteArray sessionWavBuffer; 
    
    QStringList m_translations;
    QStringList m_savedSessions;
    QString m_status = "Ready";
    QString m_petName = "Subject A";
    
    bool m_isRecording = false;
    bool m_isPlaying = false;
    bool m_saveRawWav = false;
    bool m_isCalibrating = false;
    double m_sensitivity = 1.0;
    
    Species m_species = GuineaPig;
    int sampleRate = PREFERRED_SAMPLE_RATES.last();
    std::vector<double> noiseFloorProfile;
    QJsonArray currentSessionData;
    std::vector<double> recentSamples;
};

CaviaAnalyzer::CaviaAnalyzer(QObject* parent) : QObject(parent) {
    mlClassifier = new MLClassifier(this);
    noiseFloorProfile.resize(FFT_SIZE / 2, 0.0);
}

CaviaAnalyzer::~CaviaAnalyzer() {
    if (m_isRecording) stopRecording();
    if (m_synthSink) m_synthSink->deleteLater();
    if (m_synthBuffer) m_synthBuffer->deleteLater();
}

void CaviaAnalyzer::setCurrentSpecies(int idx) { m_species = static_cast<Species>(idx); emit currentSpeciesChanged(); }
void CaviaAnalyzer::setPetName(const QString& name) { m_petName = name; emit petNameChanged(); }
void CaviaAnalyzer::setSensitivity(double val) { m_sensitivity = val; emit sensitivityChanged(); }
void CaviaAnalyzer::setSaveRawWav(bool val) { m_saveRawWav = val; emit saveRawWavChanged(); }

void CaviaAnalyzer::toggleRecording() {
    if (m_isRecording) stopRecording();
    else startRecording();
}

void CaviaAnalyzer::calibrateNoiseFloor() {
    if (!m_isRecording) {
        m_status = "Must be recording to calibrate.";
        emit statusChanged();
        return;
    }
    m_isCalibrating = true;
    std::fill(noiseFloorProfile.begin(), noiseFloorProfile.end(), 0.0);
    m_status = "Calibrating room acoustics... (3s)";
    emit statusChanged();
    
    QTimer::singleShot(3000, this, [this]() {
        m_isCalibrating = false;
        m_status = "Calibration complete. Listening...";
        emit statusChanged();
    });
}

void CaviaAnalyzer::startRecording() {
    if (m_isRecording) return;

    QMicrophonePermission micPermission;
    auto permissionStatus = qApp->checkPermission(micPermission);

    if (permissionStatus == Qt::PermissionStatus::Undetermined) {
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
    sessionWavBuffer.clear();
    if (spectrogram) spectrogram->clear();
    audioBuffer.clear();
    emit translationsChanged();

    auto devices = QMediaDevices::audioInputs();
    if (devices.isEmpty()) {
        m_status = "No microphone found!";
        emit statusChanged();
        return;
    }
    QAudioDevice device = devices.first();

    QAudioFormat format;
    format.setChannelCount(1);
    format.setSampleFormat(QAudioFormat::Int16);
    
    bool formatFound = false;
    for (int sr : PREFERRED_SAMPLE_RATES) {
        format.setSampleRate(sr);
        if (device.isFormatSupported(format)) {
            sampleRate = sr;
            formatFound = true;
            break;
        }
    }

    if (!formatFound) {
        m_status = "No supported audio formats found.";
        emit statusChanged();
        return;
    }

    audioSource = new QAudioSource(device, format, this);
    audioDevice = audioSource->start();

    if (!audioDevice) {
        m_status = "Failed to start recording";
        emit statusChanged();
        return;
    }

    m_isRecording = true;
    m_status = QString("Listening on %1 kHz").arg(sampleRate / 1000.0);
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
    if (m_saveRawWav) sessionWavBuffer.append(chunk); 
    
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

double CaviaAnalyzer::computeAutocorrelationPulseRate(const std::vector<double>& waveformSamples, int sr) {
    if (waveformSamples.size() < 100) return 0.0;

    std::vector<double> envelope(waveformSamples.size());
    for (size_t i = 0; i < waveformSamples.size(); ++i) {
        envelope[i] = std::abs(waveformSamples[i]);
    }
    
    std::vector<double> autocorr(envelope.size() / 2, 0.0);
    for (size_t lag = 1; lag < envelope.size() / 2; ++lag) {
        for (size_t i = 0; i < envelope.size() - lag; ++i) {
            autocorr[lag] += envelope[i] * envelope[i + lag];
        }
    }

    double maxVal = 0;
    int bestLag = 0;
    for (size_t lag = sr / 100; lag < autocorr.size(); ++lag) { 
        if (autocorr[lag] > maxVal) {
            maxVal = autocorr[lag];
            bestLag = lag;
        }
    }

    if (bestLag == 0) return 0.0;
    return static_cast<double>(sr) / bestLag;
}

void CaviaAnalyzer::analyzeFrame(const int16_t* data, int offset, int sr) {
    if (m_isPlaying) return;

    // Send data to the UI Waveform 
    std::vector<double> wf(WAVEFORM_SAMPLES);
    for (int i = 0; i < WAVEFORM_SAMPLES; ++i) {
        int idx = i * (FFT_SIZE / WAVEFORM_SAMPLES);
        wf[i] = data[offset + idx] / 32768.0;
    }
    if (waveform) waveform->updateWaveform(wf);

    std::vector<std::complex<double>> frame(FFT_SIZE);
    for (int i = 0; i < FFT_SIZE; ++i) {
        double window = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (FFT_SIZE - 1.0)));
        frame[i] = std::complex<double>(data[offset + i] * window, 0.0);
    }

    std::vector<std::complex<double>> fftData = frame;
    FFT::forward(fftData);

    std::vector<double> mags(FFT_SIZE / 2);
    double maxMag = 1e-8;
    double sumMag = 0;
    double peakFreq = 0;
    double minFreq = (m_species == Capybara) ? 50.0 : 100.0;

    for (int k = 0; k < FFT_SIZE / 2; ++k) {
        double freq = k * static_cast<double>(sr) / FFT_SIZE;
        if (freq < minFreq || freq > 50000) { 
            mags[k] = 0; continue; 
        }
        
        double mag = std::abs(fftData[k]);
        
        if (m_isCalibrating) {
            noiseFloorProfile[k] = (noiseFloorProfile[k] + mag) / 2.0;
        } else {
            mag = std::max(0.0, mag - noiseFloorProfile[k]); 
        }
        
        mags[k] = mag;
        sumMag += mag;
        if (mag > maxMag) { maxMag = mag; peakFreq = freq; }
    }

    if (m_isCalibrating) return;

    double avgMag = sumMag / (FFT_SIZE / 2);
    for (double& m : mags) m = std::min(1.0, m / (maxMag * 1.5));
    if (spectrogram) spectrogram->addColumn(mags);

    static bool inCall = false;
    static double callStart = 0;
    static double freqSum = 0;
    static int callFrames = 0;

    double baseThreshold = 14000.0 * (1.0 / m_sensitivity);
    bool strongSignal = (maxMag > baseThreshold) && (maxMag > avgMag * (6.0 / m_sensitivity));
    double currentTime = offset / static_cast<double>(sr);

    if (strongSignal) {
        if (!inCall) {
            inCall = true;
            callStart = currentTime;
            freqSum = 0;
            callFrames = 0;
            recentSamples.clear(); 
        }
        freqSum += peakFreq;
        callFrames++;
        
        for(int i = 0; i < FFT_SIZE; i++) {
            recentSamples.push_back(data[offset + i] / 32768.0);
        }
        
    } else if (inCall) {
        inCall = false;
        double duration = (callFrames * HOP_SIZE) / static_cast<double>(sr);
        double avgFreq = freqSum / callFrames;
        
        double pulseRate = computeAutocorrelationPulseRate(recentSamples, sr);
        classifyCall(avgFreq, duration, callStart, maxMag, pulseRate);
    }
}

void CaviaAnalyzer::classifyCall(double freq, double duration, double timestamp, double maxMag, double pulseRate) {
    if (duration < 0.02) return;
    QString speciesStr = (m_species == GuineaPig) ? "Guinea Pig" : "Capybara";
    std::vector<double> currentPulseRates; 
    QString mlPrediction = mlClassifier->predict(freq, duration, maxMag, currentPulseRates, speciesStr);

    QString meaning;
    
    if (m_species == GuineaPig) {
        if (pulseRate >= 13.0 && pulseRate <= 25.0 && duration >= 0.5) {
            QString ageStr = "Adult";
            if (freq > 400) ageStr = "Pup (<10 days)";
            else if (freq > 300) ageStr = "Adolescent";
            meaning = QString("Purr / Rumble Strutting [%1]\n> Pulse Rate: %2 Hz").arg(ageStr).arg(pulseRate, 0, 'f', 1);
        }
        else if (freq >= 400 && freq <= 500 && duration >= 0.05 && duration <= 0.2) meaning = "Chutter\n> Exploration / General Comfort";
        else if (freq >= 500 && freq <= 3500 && duration >= 0.25 && duration <= 1.0) meaning = "Wheek / Whistle\n> Food Anticipation / Excitement";
        else if (freq >= 500 && freq <= 1500 && duration >= 0.25 && duration <= 0.5) meaning = "Squeal\n> Minor Pain / Social Dispute";
        else if (freq >= 800 && duration >= 0.5 && duration <= 0.7) meaning = "Scream / Shriek\n> Extreme Fear / Predator Alarm";
        else if (freq >= 4500 && freq <= 6000 && duration < 0.1) meaning = "Chirp (Bird-song)\n> Deep Calming / Rare";
        else if (freq >= 1000 && freq <= 4000 && duration < 0.1) meaning = "Chirrup\n> Aerial Predator Alarm";
        else if (freq >= 100 && freq <= 1000 && duration >= 0.1 && maxMag > 15000) meaning = "Teeth Chattering\n> Warning / Aggression (Back off)";
        else if (freq >= 200 && freq <= 300 && duration > 1.0) meaning = "Whining / Moaning\n> Discomfort / Mild Fear";
        else if (freq >= 100 && freq <= 250 && duration > 0.05 && maxMag < 18000) meaning = "Bubbling\n> Deep Relaxation / Bonding";
        else if (freq >= 3000 && freq <= 5000 && duration > 0.5) meaning = "Medical Warning: Clicking / Wheezing\n> Possible Respiratory Distress (Seek Vet)";
    } else { 
        if (freq >= 50 && freq <= 150 && duration < 0.1) meaning = "Click / Purr\n> Contact / Spatial Monitoring";
        else if (freq >= 200 && freq <= 600 && duration > 0.08 && duration < 0.16) meaning = "Bark\n> Predator Alarm / Startle";
        else if (duration >= 1.0 && duration <= 2.0 && freq >= 1500) meaning = "Whistle\n> Pup Isolation / Distress";
        else if (freq >= 500 && duration >= 0.2 && duration <= 0.7) meaning = "Whine\n> Begging / Appeasement";
        else if (freq > 20000) meaning = "USV Squeal\n> Extreme Panic / Restraint";
    }

    if (!meaning.isEmpty()) {
        QString entry = QString::asprintf("[%.1fs] %s\n> Freq: %.0f Hz | Dur: %.2f s", timestamp, qPrintable(meaning), freq, duration);
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
    if (currentSessionData.isEmpty()) {
        m_status = "No calls to save!";
        emit statusChanged();
        return;
    }
    
    QString timestampStr = QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss");
    QString basePath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/CaviaBioacoustics";
    QDir().mkpath(basePath);
    
    QString jsonFilename = basePath + "/session_" + m_petName + "_" + timestampStr + ".json";
    QJsonObject root;
    root["subject"] = m_petName;
    root["species"] = (m_species == GuineaPig) ? "Guinea Pig" : "Capybara";
    root["date"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    root["calls"] = currentSessionData;

    QFile file(jsonFilename);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    }
    
    if (m_saveRawWav && !sessionWavBuffer.isEmpty()) {
        writeWavFile(basePath + "/raw_audio_" + m_petName + "_" + timestampStr + ".wav");
    }

    m_savedSessions.prepend(m_petName + " — " + QDateTime::currentDateTime().toString("MM-dd hh:mm") + 
                            " (" + QString::number(currentSessionData.size()) + " calls)");
    emit savedSessionsChanged();
    
    m_status = "Session Saved Successfully!";
    emit statusChanged();
}

void CaviaAnalyzer::writeWavFile(const QString& filename) {
    QFile wavFile(filename);
    if (wavFile.open(QIODevice::WriteOnly)) {
        QDataStream out(&wavFile);
        out.setByteOrder(QDataStream::LittleEndian);
        
        out.writeRawData("RIFF", 4);
        out << static_cast<quint32>(36 + sessionWavBuffer.size());
        out.writeRawData("WAVE", 4);
        out.writeRawData("fmt ", 4);
        out << static_cast<quint32>(16); 
        out << static_cast<quint16>(1);  
        out << static_cast<quint16>(1);  
        out << static_cast<quint32>(sampleRate); 
        out << static_cast<quint32>(sampleRate * 1 * 2);
        out << static_cast<quint16>(2);  
        out << static_cast<quint16>(16); 
        out.writeRawData("data", 4);
        out << static_cast<quint32>(sessionWavBuffer.size());
        out.writeRawData(sessionWavBuffer.constData(), sessionWavBuffer.size());
    }
}

void CaviaAnalyzer::clearCurrentSession() {
    m_translations.clear();
    currentSessionData = QJsonArray();
    sessionWavBuffer.clear();
    emit translationsChanged();
}

void CaviaAnalyzer::playCall(const QString& callType) {
    if (m_synthSink) {
        m_synthSink->stop();
        m_synthSink->deleteLater();
        m_synthBuffer->deleteLater();
    }
    m_synthData.clear();
    int sr = 48000;
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
            if (hasHarmonics) { wave += 0.5 * std::sin(2 * phase); wave += 0.25 * std::sin(3 * phase); }
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
        for (int i = 0; i < samples; ++i) m_synthData.append(reinterpret_cast<const char*>(&zero), sizeof(int16_t));
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
    } else if (callType == "Capy Click") {
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

// -----------------------------------------------------------------
// UPGRADED QML FRONTEND (3 Tabs + Margins + Settings + Waveform)
// -----------------------------------------------------------------
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
        anchors.topMargin: 50 
        anchors.leftMargin: 16
        anchors.rightMargin: 16
        anchors.bottomMargin: 16
        spacing: 12

        RowLayout {
            spacing: 12
            Text { text: "Species:"; color: "#8b949e"; font.pixelSize: 16; font.bold: true }
            ComboBox {
                id: speciesCombo
                Layout.fillWidth: true
                height: 44
                model: ["Guinea Pig", "Capybara"]
                currentIndex: backend.currentSpecies
                onCurrentIndexChanged: backend.currentSpecies = currentIndex
                enabled: !backend.isRecording
                contentItem: Text {
                    text: speciesCombo.displayText
                    color: "white"
                    font.pixelSize: 16
                    font.bold: true
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignHCenter
                }
                background: Rectangle { color: "#21262d"; border.color: speciesCombo.focus ? "#58a6ff" : "#30363d"; border.width: 2; radius: 8 }
            }
        }

        Button {
            Layout.fillWidth: true
            height: 56
            text: backend.isRecording ? "STOP RECORDING" : "START RECORDING"
            font.bold: true; font.pixelSize: 18
            onClicked: backend.toggleRecording()
            background: Rectangle { 
                color: backend.isRecording ? "#da3633" : "#238636"
                radius: 8; border.color: backend.isRecording ? "#b62324" : "#2ea043"; border.width: 1
            }
            contentItem: Text { text: parent.text; color: "white"; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter; font: parent.font }
        }

        Text { text: backend.status; color: "#58a6ff"; font.pixelSize: 13; font.italic: true; Layout.alignment: Qt.AlignHCenter }

        ColumnLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 220
            spacing: 6

            // Top Box: The Live Waveform
            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: "black"
                border.color: "#30363d"
                border.width: 2
                radius: 4
                clip: true
                Waveform { id: wf; anchors.fill: parent; anchors.margins: 2; objectName: "waveformItem" }
            }

            // Bottom Box: The Spectrogram
            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: "black"
                border.color: "#30363d"
                border.width: 2
                radius: 4
                clip: true
                Spectrogram { id: spectro; anchors.fill: parent; anchors.margins: 2; objectName: "spectroItem" }
            }
        }

        RowLayout {
            spacing: 12
            Button { 
                text: "Save Session"
                Layout.fillWidth: true; height: 36
                onClicked: backend.saveCurrentSession()
                background: Rectangle { color: "#21262d"; radius: 6; border.color: "#30363d" }
                contentItem: Text { text: parent.text; color: "#c9d1d9"; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
            }
            Button { 
                text: "Clear Data"
                Layout.fillWidth: true; height: 36
                onClicked: backend.clearCurrentSession() 
                background: Rectangle { color: "#21262d"; radius: 6; border.color: "#30363d" }
                contentItem: Text { text: parent.text; color: "#c9d1d9"; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
            }
        }

        TabBar {
            id: bar
            Layout.fillWidth: true
            background: Rectangle { color: "transparent" }
            TabButton { 
                text: "Translator" 
                contentItem: Text { text: parent.text; color: bar.currentIndex === 0 ? "white" : "#8b949e"; horizontalAlignment: Text.AlignHCenter; font.bold: true }
                background: Rectangle { color: bar.currentIndex === 0 ? "#21262d" : "transparent"; radius: 4 }
            }
            TabButton { 
                text: "Logs" 
                contentItem: Text { text: parent.text; color: bar.currentIndex === 1 ? "white" : "#8b949e"; horizontalAlignment: Text.AlignHCenter; font.bold: true }
                background: Rectangle { color: bar.currentIndex === 1 ? "#21262d" : "transparent"; radius: 4 }
            }
            TabButton { 
                text: "Settings" 
                contentItem: Text { text: parent.text; color: bar.currentIndex === 2 ? "white" : "#8b949e"; horizontalAlignment: Text.AlignHCenter; font.bold: true }
                background: Rectangle { color: bar.currentIndex === 2 ? "#21262d" : "transparent"; radius: 4 }
            }
        }

        StackLayout {
            currentIndex: bar.currentIndex
            Layout.fillWidth: true
            Layout.fillHeight: true

            // TAB 0: TRANSLATOR
            Rectangle {
                color: "#161b22"; radius: 8; border.color: "#30363d"
                ColumnLayout {
                    anchors.fill: parent; anchors.margins: 12
                    ListView {
                        Layout.fillHeight: true; Layout.fillWidth: true
                        clip: true; spacing: 8; model: backend.translations
                        delegate: Text { text: modelData; color: "#3fb950"; font.pixelSize: 15; wrapMode: Text.Wrap; width: parent.width }
                    }
                    Text { text: "Synthesize Warning (No Spatial Cues)"; color: "#8b949e"; font.pixelSize: 12; font.bold: true; Layout.topMargin: 8 }
                    Flow {
                        Layout.fillWidth: true; spacing: 8
                        Repeater {
                            model: backend.currentSpecies === 0 ? 
                                [
                                    {c: "Purr", m: "Purr (Content)"},
                                    {c: "Chutter", m: "Chutter (Explore)"},
                                    {c: "Wheek", m: "Wheek (Excite)"},
                                    {c: "Squeal", m: "Squeal (Dispute)"},
                                    {c: "Scream", m: "Scream (Alarm)"},
                                    {c: "Chirp", m: "Chirp (Calm)"},
                                    {c: "Chirrup", m: "Chirrup (Warning)"},
                                    {c: "Tooth-Chatter", m: "Chatter (Aggression)"}
                                ] : 
                                [
                                    {c: "Capy Click", m: "Click (Contact)"},
                                    {c: "Capy Bark", m: "Bark (Alarm)"},
                                    {c: "Capy Whistle", m: "Whistle (Distress)"},
                                    {c: "Capy Whine", m: "Whine (Appease)"},
                                    {c: "Tooth-Chatter", m: "Chatter (Threat)"}
                                ]
                            Button {
                                text: modelData.m; onClicked: backend.playCall(modelData.c)
                                background: Rectangle { color: "#21262d"; radius: 12; border.color: "#8b949e" }
                                contentItem: Text { text: parent.text; color: "#58a6ff"; padding: 6; font.pixelSize: 12 }
                            }
                        }
                    }
                }
            }

            // TAB 1: LOGS
            Rectangle {
                color: "#161b22"; radius: 8; border.color: "#30363d"
                ListView {
                    anchors.fill: parent; anchors.margins: 12; clip: true; spacing: 8
                    model: backend.savedSessions
                    delegate: ItemDelegate {
                        width: parent.width; height: 40
                        text: modelData
                        contentItem: Text { text: parent.text; color: "#8b949e"; font.pixelSize: 14 }
                        background: Rectangle { color: parent.hovered ? "#30363d" : "transparent"; radius: 4 }
                        onClicked: { backend.status = "Opened " + modelData; }
                    }
                }
            }

            // TAB 2: SETTINGS
            Rectangle {
                color: "#161b22"; radius: 8; border.color: "#30363d"
                ScrollView {
                    anchors.fill: parent; anchors.margins: 16; clip: true
                    ColumnLayout {
                        width: parent.width; spacing: 16
                        
                        Text { text: "Subject / Pet Name"; color: "white"; font.bold: true }
                        TextField {
                            Layout.fillWidth: true; text: backend.petName
                            onTextChanged: backend.petName = text
                            color: "white"
                            background: Rectangle { color: "#0d1117"; border.color: "#30363d"; radius: 4; implicitHeight: 40 }
                        }
                        
                        Text { text: "Mic Sensitivity (Left=Noisy Room, Right=Quiet)"; color: "white"; font.bold: true; Layout.topMargin: 8 }
                        Slider {
                            Layout.fillWidth: true; from: 0.2; to: 3.0; value: backend.sensitivity
                            onValueChanged: backend.sensitivity = value
                        }
                        
                        RowLayout {
                            Text { text: "Export Raw .WAV Audio"; color: "white"; font.bold: true; Layout.fillWidth: true }
                            Switch { checked: backend.saveRawWav; onCheckedChanged: backend.saveRawWav = checked }
                        }

                        Button {
                            Layout.fillWidth: true; height: 44
                            text: "Calibrate Background Noise"
                            onClicked: backend.calibrateNoiseFloor()
                            background: Rectangle { color: "#21262d"; radius: 6; border.color: "#58a6ff" }
                            contentItem: Text { text: parent.text; color: "#58a6ff"; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                        }
                    }
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
    qmlRegisterType<WaveformItem>("Bioacoustics", 1, 0, "Waveform");
    
    QQmlApplicationEngine engine;
    CaviaAnalyzer backend;
    engine.rootContext()->setContextProperty("backend", &backend);
    engine.loadData(qmlData);
    
    if (engine.rootObjects().isEmpty()) return -1;
    
    auto root = engine.rootObjects().first();
    if (auto* spectro = root->findChild<SpectrogramItem*>("spectroItem")) {
        backend.setSpectrogram(spectro);
    }
    if (auto* wf = root->findChild<WaveformItem*>("waveformItem")) {
        backend.setWaveform(wf);
    }
    
    return app.exec();
}
#include "main.moc"
