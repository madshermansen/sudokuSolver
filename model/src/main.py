from train.sudokuNet import sudokuNet


def main():
    sudokuSolverModel = sudokuNet()
    sudokuSolverModel.train()
    sudokuSolverModel.load_model("output/20241109-150345.keras")


if __name__ == "__main__":
    main()
