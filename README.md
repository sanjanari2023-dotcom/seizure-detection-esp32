# Real-Time Seizure Detection and Emergency Alert Device

Real-time seizure detection and alert prototype using ESP32, biomedical sensors and machine learning. Major project (team of 4), SJB Institute of Technology.

## Hardware
- ESP32
- MAX30102: heart rate and SpO2
- MPU6050: motion and fall detection
- BioAmp EXG Pill with ADS1115 (16-bit ADC): bioelectric signal acquisition
- GSM module: SMS alerts

## Features
- Live dashboard with real-time waveforms (PySide6, PyQtGraph, PySerial)
- Random Forest classifier on extracted signal features
- Sensor fusion with heart rate and SpO2
- Heart-rate alerts and gyroscope-based fall detection
- GSM SMS alerts to caregivers
- CHB-MIT EEG recordings processed with MNE-Python

## Files
- ESP32_code.ino: ESP32 Arduino firmware
- dashboard.py: Python dashboard and ML code

## Tools
Arduino IDE, Python, scikit-learn, NumPy, MNE-Python

Note: this is a prototype and not a certified medical device.
