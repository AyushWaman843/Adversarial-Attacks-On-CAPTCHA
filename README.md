# Adversarial Attacks on CAPTCHA Recognition Systems

A research project exploring the vulnerability of CAPTCHA recognition models to adversarial attacks. This project implements and compares multiple adversarial attack methods to evaluate CAPTCHA security.

## 📋 Overview

This project investigates how adversarial examples can fool AI-based CAPTCHA solvers while remaining recognizable to humans. We implement several state-of-the-art attack techniques and analyze their effectiveness against trained CAPTCHA recognition models.

## 🎯 Features

- **CAPTCHA Recognition Model**: Deep learning model trained to solve text-based CAPTCHAs
- **Multiple Attack Methods**:
  - FGSM (Fast Gradient Sign Method)
  - PGD (Projected Gradient Descent)
  - Carlini-Wagner (C&W) Attack
  - MNIST-based adversarial examples
- **Comprehensive Analysis**:
  - Accuracy vs Confidence plots
  - Confusion matrices
  - Attack success rate visualization
  - Side-by-side comparison of original vs adversarial images

## 🛠️ Technologies Used

- Python 3.x
- TensorFlow/Keras
- PyTorch
- NumPy
- Matplotlib
- OpenCV

## 📁 Project Structure
```
├── app.py, app1.py, app2.py, app3.py  # Web applications
├── captcha_model.h5                    # Trained Keras model
├── model.pth                           # Trained PyTorch model
├── label_encoder.pkl                   # Label encoding for predictions
├── attack_results/                     # Generated adversarial examples
├── data/                               # Training data
├── samples/                            # Sample CAPTCHA images
├── *.ipynb                             # Jupyter notebooks for experiments
└── Project doc.docx                    # Project documentation
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/AyushWaman843/Adversarial-Attacks-On-CAPTCHA.git
cd Adversarial-Attacks-On-CAPTCHA
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Run Web Application
```bash
python app.py
```

### Run Adversarial Attacks
```bash
# FGSM Attack
python adv_attk_MNIST.ipynb

# Carlini-Wagner Attack
python cw_attack_accuracy_summary.png

# PGD Attack
python pgd_examples.png
```

### Test CAPTCHA Recognition
```bash
python test.ipynb
```

## 📊 Results

The project generates various visualizations:
- **Confusion Matrices**: Show misclassification patterns
- **Accuracy Plots**: Compare model performance on clean vs adversarial examples
- **Example Comparisons**: Visual side-by-side of original and attacked CAPTCHAs

## 🔬 Attack Methods Explained

### FGSM (Fast Gradient Sign Method)
Fast single-step attack that adds noise in the direction of the gradient.

### PGD (Projected Gradient Descent)
Iterative attack that takes multiple small steps to create adversarial examples.

### Carlini-Wagner (C&W)
Optimization-based attack that minimizes perturbation while maximizing misclassification.

## 📈 Key Findings

- Adversarial attacks can significantly reduce CAPTCHA recognition accuracy
- C&W attacks produce more subtle perturbations than FGSM
- Trade-off between attack strength and image quality
- Human readability remains intact even with successful attacks

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is for educational and research purposes only.

## 👤 Author

**Ayush Waman**
- GitHub: [@AyushWaman843](https://github.com/AyushWaman843)

## 🙏 Acknowledgments

- Research papers on adversarial machine learning
- CAPTCHA datasets and benchmarks
- Open-source deep learning community

## ⚠️ Disclaimer

This project is intended for educational and research purposes to improve CAPTCHA security. Do not use these techniques for malicious purposes or to bypass security systems without authorization.

---

**Note**: This project demonstrates security vulnerabilities in automated CAPTCHA solvers and aims to contribute to the development of more robust authentication systems.
