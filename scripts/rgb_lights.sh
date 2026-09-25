#!/bin/bash
# Turn on both ASUS 5090 RGB lights (static white). Requires sudo for i2c access.
sudo /opt/openrgb/OpenRGB.AppImage --appimage-extract-and-run -d 0,1 -m static -c FFFFFF
