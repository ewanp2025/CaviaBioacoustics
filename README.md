# Cavia Bioacoustics

Work In Progress

A realtime audio analysis mobile app for classifying and translating the vocalisations of domestic Guinea Pigs (*Cavia porcellus*) and Capybaras (*Hydrochoerus hydrochaeris*).

## Features
* **Live Translation:** Automatically detects and classifies audible calls (wheeks, chhutters, barks).
* **Real-Time Spectrogram:** Visualises audio frequencies directly on your screen using custom FFT processing.
* **100% Private & Offline:** All audio analysis is performed locally on your device. No recordings or data are ever saved or transmitted to the cloud.

## Tech
* **Language:** C++17
* **Framework:** Qt 6.7 (Qt Quick, Qt Multimedia)
* **UI:** QML
* **Target OS:** Android (API 28 - 35)

## Build Instructions
This project is configured with GitHub Actions to automatically compile Release APKs and AABs. 

To build locally:
1. Ensure **Qt 6.7+** and the **Android SDK/NDK** are installed.
2. Open the `CMakeLists.txt` in Qt Creator.
3. Configure the project for an Android ARM64 kit and build.

## License
This project is free software licensed under the **GNU General Public License v3.0** (GPLv3). See the `LICENSE` file for full details.
