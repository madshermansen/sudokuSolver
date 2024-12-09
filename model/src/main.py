from train.sudokuNet import sudokuNet
from server.api import run_server
import os
import datetime

def main():
    sudokuSolverModel = sudokuNet()
    modelURL = '1209-201103-epochs50.keras'

    sudokuSolverModel.load_model(f'output/model/{modelURL}')

    # # run server with model
    run_server(sudokuSolverModel)
    # train(sudokuSolverModel)
    # sudokuSolverModel.evaluate()

def test_predict(model):
    # loop through predict folder
    folder_path = 'data/predict/'

    for filename in os.listdir(folder_path):
        print(f"Predicting {filename}")
        model.predict(folder_path + filename)

    print("/n The evvaluation of the model is:")
    model.evaluate()
    

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
