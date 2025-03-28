# Sign Language Recognition and Translation

This project aims to develop a system that recognizes sign language gestures using object detection techniques and translates them into text in real time. The solution combines a React-based frontend, a FastAPI backend, and a PyTorch-based machine learning model for gesture recognition.

---

## Features

- **Real-time Sign Language Detection**: Utilize object detection to identify hand gestures representing sign language.
- **Text Translation**: Convert recognized gestures into corresponding textual representation.
- **Interactive Interface**: A React-based frontend provides an intuitive user experience.
- **Efficient Backend**: A FastAPI backend handles API requests and coordinates the translation process.
- **Powerful Machine Learning**: PyTorch is used for training and deploying the gesture recognition model.

---

## Tech Stack

### Frontend

- **React**
  - Provides a responsive and dynamic user interface for interaction with the system.

### Backend

- **FastAPI**

  - Manages communication between the frontend and the machine learning model.
  - Handles user requests and serves predictions.

### Machine Learning

- **PyTorch**
  - Used to train a neural network model for object detection and gesture classification.
  - Supports real-time inference to translate gestures into text.

---

## Getting Started

### Prerequisites

All requirements are installed in the docker container and no extra action is needed.

### Installation

1. **Clone the repository**:

   ```bash
   git clone ssh://git@gitlab.informatik.hs-augsburg.de:2222/elias.haggenmueller/asl_detection.git
   cd asl_detection
   ```

2. **Rebuild container**:

   - maybe container extension is needed

---

## Usage

1.Backend:

```shell
poetry run uvicorn backend.main:app --reload
```

Frontend:

```shell
cd frontend && poetry run npm start
```

If you get an error claiming that react-scripts.sh was not found, this is due to Docker caching issues.
Run the following:

```shell
cd frontend && poetry run npm install
```

And then start the frontend as usual.

## Managing dependencies

This project is using [poetry](https://python-poetry.org/) to manage dependencies.
Compared to the usual pip, it provides better management of transitive dependencies.

You install new packages via:

```shell
poetry add <package-name>
```

Executing specific packages

```shell
poetry run <package-name>
```

You sync your installed packages and your
pyproject.toml with:

```shell
poetry install
```

(this is initially done by the dev container setup.)

---

## Folder Structure --> to change

```
.
├── frontend
│   ├── src
│   └── public
├── backend
│   ├── main.py
│   ├── ml_model
│   └── requirements.txt
└── README.md
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push to your fork.
4. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- Inspired by advancements in object detection and natural language processing.
- Thanks to the open-source community for the tools and resources.

<pre>
                                                   
_____       ________________        ____________   
\    \     /    /\          \      /            \  
 \    |   |    /  \    /\    \    |\___/\  \\___/| 
  \    \ /    /    |   \_\    |    \|____\  \___|/ 
   \    |    /     |      ___/           |  |      
   /    |    \     |      \  ____   __  /   / __   
  /    /|\    \   /     /\ \/    \ /  \/   /_/  |  
 |____|/ \|____| /_____/ |\______||____________/|  
 |    |   |    | |     | | |     ||           | /  
 |____|   |____| |_____|/ \|_____||___________|/   
                                                   
</pre>
