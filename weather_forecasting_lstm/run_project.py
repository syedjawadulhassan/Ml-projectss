import os

print("Training...")
os.system("python src/train_lstm.py")

print("Evaluating...")
os.system("python src/evaluate.py")

print("Launching dashboard...")
os.system("streamlit run app/app.py")