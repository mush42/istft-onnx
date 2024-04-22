## Create an ONNX model that performs ISTFT

If you want to stay with the ONNX stack, and you don't want to pull-in external dependencies for ISTFT, then this is for you.

Note that this is at least 4x slower than `torch` istft implementation, but the ISTFT overhead is negligible anyways.

## Usage

Install requirements:

```bash
pip3 install -r requirements.txt
```

Run the script:

```python
python3 istft_onnx
```

Make sure to correctly specify your `ISTFT` parameters.

You should also specify the maximom number of frames the exported model will operate on by specifying `--max-frames` parameters. By default it is set to 5200 (around 1 second for 22.05KHz sample rate).

The model is designed around the output of `Vocos` vocoder. Please change the inputs based on your needs. Most likely, you need to input mag and phase.

## License

See the source file for more details.
