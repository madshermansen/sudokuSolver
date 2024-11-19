from train.sudokuNet import sudokuNet
import os
import datetime


def main():
    sudokuSolverModel = sudokuNet()
    modelURL = '1112-135907-epochs10.keras'

    # sudokuSolverModel.load_model(f'output/model/{modelURL}')
    train(sudokuSolverModel)

    # loop through predict folder
    folder_path = 'data/predict/'

    for filename in os.listdir(folder_path):
        print(f"Predicting {filename}")
        sudokuSolverModel.predict(folder_path + filename)

    # Plot the training history
    print("/n The evvaluation of the model is:")
    sudokuSolverModel.evaluate()


def train(model):
    start = datetime.datetime.now()
    model.train(epochs=50, batch_size=128)

    model.get_summary()
    # Total time taken
    print(f"Total time taken: {datetime.datetime.now() - start}")


if __name__ == "__main__":
    # check if running from model directory
    if not (os.getcwd().split('/')[-1] == 'model'):
        print('==' * 24, '\nPlease switch directory to model directory\n',
              '==' * 24, sep='')
        exit()

    # run main function
    main()
