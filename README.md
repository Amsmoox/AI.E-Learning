# AI.E-Learning
### Documentation for the Adaptive Learning Specialist E-Learning Platform

#### Overview
The Adaptive Learning Specialist E-Learning Platform is a sophisticated system designed to personalize student learning paths using advanced AI techniques, particularly Reinforcement Learning (RL). The platform adapts content and activities based on student performance to optimize learning outcomes.

#### Key Features
- **Personalized Learning Paths**: Utilizes Reinforcement Learning algorithms to dynamically adjust the sequence of learning materials based on individual student performance.
- **Backend and AI Development**: The platform’s backend is developed using Django, with integration of AI models using TensorFlow and Keras.
- **Task Management**: Celery is used to manage background tasks efficiently.
- **Data Storage and Retrieval**: PostgreSQL is employed for relational data storage, with ElasticSearch providing advanced search capabilities.
- **Security**: API access is secured using OAuth 2.0 and JWT tokens.
- **Scalability**: Docker and Kubernetes are used for deployment, ensuring the system can scale and maintain reliability.

#### Demo Code
The code provided in this repository is a demonstration of the system. The actual production code cannot be shared due to restrictions. The demo illustrates the main functionalities, including personalized learning path generation, Q-learning, and Case-Based Reasoning (CBR) integration.

#### Technologies Used
- **Backend**: Django, Python
- **AI & Machine Learning**: TensorFlow, Keras, Reinforcement Learning, Pandas, Numpy, Scipy
- **Task Management**: Celery
- **Data Storage**: PostgreSQL, Redis
- **Search**: ElasticSearch
- **Data Streaming**: Kafka
- **Security**: OAuth 2.0, JWT
- **Deployment**: Docker, Kubernetes
- **Visualization**: Matplotlib, Plotly
- **Data Formats**: JSON, Excel

#### How the System Works
1. **Pretest Evaluation**: Students undergo a pretest to assess their initial knowledge level.
2. **Learning Path Assignment**: Based on the pretest results, students are assigned a personalized learning path using a combination of Q-learning and Case-Based Reasoning (CBR).
3. **Learning Sessions**: Students go through a series of sessions with varied content, dynamically adjusted to maximize learning efficiency.
4. **Quiz Assessments**: After each session, students take quizzes to measure their progress.
5. **Continuous Adaptation**: The system continuously adapts the learning paths based on the quiz scores and learning patterns.

#### Running the Demo
To run the demo, follow these steps:
1. **Clone the Repository**: `git clone https://github.com/Amsmoox/AI.E-Learning.git`
2. **Install Dependencies**: Ensure you have Python installed and run `pip install -r requirements.txt` (comming soon).
3. **Run the Simulation**: Execute the provided scripts to see how the system assigns learning paths and processes student data.
   - `Generate.py` will generate the initial dataset and simulate the learning paths.
   - `analyse.py` will analyze the results using Q-learning and CBR, providing insights into the personalized learning paths.

#### Contact
For any questions or support, please contact me mharrech.ayoub@gmail.com

