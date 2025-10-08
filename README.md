**dvn**: codeword for Drone Visual Navigation, this is the folder containing compute vision logic
**sat**: folder containing logic to download satelite images

### Setup
intall ffmpegh

```bash
sudo apt update && sudo apt install ffmpeg
```

### Very useful commands
Delete Zone.Identifier files. Run at project root in a bash WSL terminal.
```bash
find . -type f \( -name 'Zone.Identifier' -o -name '*.Zone.Identifier' -o -name '*:Zone.Identifier' \) -delete

```