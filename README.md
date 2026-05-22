# Cavia Bioacoustics

Work In Progress

Please note that I have paid for an Android developer account, but having now learned the hurdles one must jump through to place an app in the play store, I have given up and placed my APK here instead (which means this APK must be sideloade instead).

A realtime audio analysis mobile app for classifying and translating the vocalisations of domestic Guinea Pigs (*Cavia porcellus*) and Capybaras (*Hydrochoerus hydrochaeris*).

## Features
* **Live Translation:** Automatically detects and classifies audible calls (wheeks, chhutters, barks).
* **Real-Time Spectrogram:** Visualises audio frequencies directly on your screen using custom FFT processing.
* **100% Private & Offline:** All audio analysis is performed locally on your device. No recordings or data are ever saved or transmitted to the cloud.

## Hardware Limitations
* **The Nyquist Limit & Ultrasonic Calls:** Most standard smartphone microphones are physically limited to a capture sample rate of 44.1 kHz or 48 kHz. According to the Nyquist-Shannon sampling theorem, this means the absolute maximum frequency the app can detect is half of that rate (**~22 kHz to 24 kHz**). 
* Because Guinea Pigs and Capybaras utilise ultrasonic communication (with hearing capabilities extending up to 50 kHz), a standard phone microphone will physically miss these extreme high-frequency vocalisations.
* **Next Steps:** I am currently working on implementing support for external USB ultrasonic bioacoustics microphones (capable of 96 kHz / 192 kHz sample rates) to bypass this smartphone hardware limit.

## Tech
* **Language:** C++17
* **Framework:** Qt 6.7 (Qt Quick, Qt Multimedia)
* **UI:** QML
* **Target OS:** Android (API 28 - 35)

## Build Instructions
This project is configured with GitHub Actions to automatically compile Release APKs and AABs. 

*Note: The current GitHub Actions `.yml` workflow successfully builds both files, but it is a known issue that it only signs the `.apk`. The `.aab` (Android App Bundle) is currently output unsigned.*

To build locally:
1. Ensure **Qt 6.7+** and the **Android SDK/NDK** are installed.
2. Open the `CMakeLists.txt` in Qt Creator.
3. Configure the project for an Android ARM64 kit and build.

## License
This project is free software licensed under the **GNU General Public License v3.0** (GPLv3). See the `LICENSE` file for full details.
