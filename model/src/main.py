from train.sudokuNet import sudokuNet
import os
import datetime

def main():
    sudokuSolverModel = sudokuNet()
    modelURL = '1112-135907-epochs10.keras'

    sudokuSolverModel.load_model(f'output/model/{modelURL}')
    # train(sudokuSolverModel)

    # loop through predict folder
    folder_path = 'data/predict/'

    for filename in os.listdir(folder_path):
        print(f"Predicting {filename}")
        print(sudokuSolverModel.predict(folder_path + filename))

    # Plot the training history
    sudokuSolverModel.evaluate()


def train(model):
    start = datetime.datetime.now()
    model.train(epochs=10, batch_size=128)
    # Total time taken
    print(f"Total time taken: {datetime.datetime.now() - start}")

if __name__ == "__main__":
    main()
