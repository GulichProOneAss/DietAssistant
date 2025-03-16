# Импорт библиотек
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Создание синтетического датасета
def create_synthetic_dataset(n_samples=1000):
    np.random.seed(42)
    data = {
        'calories': np.random.uniform(10, 500, n_samples),
        'protein': np.random.uniform(0, 50, n_samples),
        'fat': np.random.uniform(0, 40, n_samples),
        'carbs': np.random.uniform(0, 100, n_samples),
        'goal': np.random.choice(['weight_loss', 'muscle_gain', 'maintenance'], n_samples)
    }
    df = pd.DataFrame(data)
    df.to_csv('synthetic_nutrition_data.csv', index=False)
    print("Синтетический датасет создан и сохранён как 'synthetic_nutrition_data.csv'")
    return df

# Загрузка данных
def load_data(file_path='synthetic_nutrition_data.csv'):
    data = pd.read_csv(file_path)
    print("Данные загружены. Размер:", data.shape)
    return data

# Подготовка данных
def preprocess_data(df):
    features = df[['calories', 'protein', 'fat', 'carbs']].values
    labels = df['goal'].map({'weight_loss': 0, 'muscle_gain': 1, 'maintenance': 2}).values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    print("Данные подготовлены:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    return X_train, X_test, y_train, y_test

# Определение модели
class DietAssistantModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, output_size=3):
        super(DietAssistantModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Обучение модели
def train_model(model, X_train, y_train, epochs=50, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

# Тестирование модели
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Сохранение модели
def save_model(model, path="model.pt"):
    torch.save(model.state_dict(), path)
    print(f"Модель сохранена в файл: {path}")

# Загрузка модели
def load_model(input_size=4, hidden_size=16, output_size=3, path="model.pt"):
    model = DietAssistantModel(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Модель загружена из файла: {path}")
    return model

# Визуализация данных
def visualize_data(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df['calories'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Распределение калорийности продуктов')
    plt.xlabel('Калории')
    plt.ylabel('Количество')
    plt.grid(True)
    plt.show()
    print("Визуализация данных завершена")

# Основная функция
def main():
    print("Проект 'Интеллектуальный помощник формирования рациона питания' запущен!")
    df = create_synthetic_dataset()
    visualize_data(df)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = DietAssistantModel()
    print("Модель создана:", model)
    print("Обучение модели...")
    train_model(model, X_train, y_train)
    print("Тестирование модели...")
    evaluate_model(model, X_test, y_test)
    print("Сохранение модели...")
    save_model(model, "model.pt")
    print("Загрузка сохранённой модели...")
    loaded_model = load_model()
    print("Тестирование загруженной модели...")
    evaluate_model(loaded_model, X_test, y_test)

if __name__ == "__main__":
    main()

# Запуск программы
print("Запускаем программу...")
main()
