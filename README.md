# morphology-segmentation
Morphology Segmentation using DPGMM

Research project done as a part of my research internship at NAAMII.

# Instructions

1. Add a raw Nepali text corpus in the `data` folder. You can select the name of the file in run.py
2. Set the parameters in run.py
3. Run run.py
    - After every 10 iterations of the Gibbs Sampling, model is saved in `models` directory.
    - You can continue the training later with a saved model.
4. Run inference.py for inference

All the mathematical background can be found [here](math_model.pdf).
