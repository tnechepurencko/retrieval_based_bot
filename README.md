# retrieval_based_bot

# How to use model.py
1. Set up a Python environment to ensure that you do not break your system. You can do this by running the following command:
    * If python3 is recognized as just python in your system then run:
        ```
        python -m venv env
        ```
    * Or if python3 is recognized as python3:
        ```
        python3 -m venv env
        ```

2. Activate the Python environment:
    * On Windows run:
        ```
        env\Scripts\activate
        ```
    * On Ubuntu run:
        ```
        env/bin/activate
        ```

3. Install Python dependencies by running the following command from the root directory of the repository:
    ```
    pip install -r requirements.txt
    ```
4. Install the embeddings file from [here](https://drive.google.com/file/d/1pNuUrR6mdsnYx-vQOKqc-H7QXFG0SZjm/view?usp=sharing)

Now you can import the module model from another file, initialize an object from the Model() class, pass the question to get_top_ans() function along with the number of answers you want it to return.

There is a commented example inside model.py that can help you.
