# sudokuSolver
uses CNN to solve sudoku problems

# Presentations

Presentations are available in the following links:

[November 13, 2024](/docs/nov13-presentation.pdf)
[December 10, 2024](/docs/dec10-presentation.pdf)

# Installation & Setup

Clone the repository using the following command:

```bash
git clone https://github.com/madshermansen/sudokuSolver.git
cd sudokuSolver
```

Install python and node dependencies
```bash
cd model
pip install -r requirements.txt

cd ../sudokuapp
npm install
```

First start the backend server
```bash
cd model
python main.py
```

Copy the server URL and change the URL in camera.tsx in /sudokuapp/src/components. Open a new Terminal in the sudokuapp directory then run the following to start the Expo app:
```bash
cd sudokuapp
npm start
```