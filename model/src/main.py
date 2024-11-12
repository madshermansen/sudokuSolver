from train.sudokuNet import sudokuNet
import os


def main():
    sudokuSolverModel = sudokuNet()
    modelURL = '1110-191428-epochs3.keras'

    sudokuSolverModel.load_model(f'output/model/{modelURL}')
    # sudokuSolverModel.train(epochs=1, batch_size=128)

    # loop through predict folder
    folder_path = 'data/predict/'

    for filename in os.listdir(folder_path):
        print(f"Predicting {filename}")
        print(sudokuSolverModel.predict(folder_path + filename))

    # Plot the training history
    sudokuSolverModel.evaluate()

if __name__ == "__main__":
    main()
