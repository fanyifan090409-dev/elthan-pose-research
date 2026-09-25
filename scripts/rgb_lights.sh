#!/bin/bash
# Both ASUS 5090 RGB: rainbow wave, full brightness. Requires sudo for i2c access.
sudo /opt/openrgb/OpenRGB.AppImage --appimage-extract-and-run -d 0 -m rainbow -b 100 -s 40
sudo /opt/openrgb/OpenRGB.AppImage --appimage-extract-and-run -d 1 -m rainbow -b 100 -s 40
