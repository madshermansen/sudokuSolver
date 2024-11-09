from train.sudokuNet import sudokuNet

import matplotlib.pyplot as plt


def main():
    sudokuSolverModel = sudokuNet()
    #sudokuSolverModel.load_model('output/model/model_20241109-155522-epochs25.keras.keras')
    sudokuSolverModel.train(epochs=25, batch_size=128)

if __name__ == "__main__":
    main()
