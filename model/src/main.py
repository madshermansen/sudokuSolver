from train.sudokuNet import sudokuNet
import os


def main():
    sudokuSolverModel = sudokuNet()
    sudokuSolverModel.load_model('output/model/1109-162626-epochs25.keras')
    #sudokuSolverModel.train(epochs=25, batch_size=128)

    # loop through predict folder
    folder_path = 'data/testSet/'

    for filename in os.listdir(folder_path):
        print(f"Predicting {filename}")
        print(sudokuSolverModel.predict(folder_path + filename))

    # Plot the training history
    sudokuSolverModel.evaluate()

if __name__ == "__main__":
    main()
