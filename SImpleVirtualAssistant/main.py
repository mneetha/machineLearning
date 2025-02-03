from datetime import datetime, timedelta

from neuralintents import BasicAssistant
import sys
import yfinance as yf

stock_tickers = ['AAPL', 'META', 'GS', 'TSLA']

todos = ['Wash car', 'Watch NeuralNine videos', 'Go shopping']

def stock_function():
    # Get today's date
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Get the date one year ago
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

    for ticker in stock_tickers:
        # Fetch historical stock data
        data = yf.download(ticker, start=start_date, end=end_date)

        # Select only the required columns
        formatted_data = data[['Close', 'High', 'Low', 'Open', 'Volume']]

        # Print with proper formatting
        print(formatted_data.head())  # Display first few rows

def greeting():
    print("in greeting")

def todo_show():
    print("Your TODO list:")
    for todo in todos:
        print(todo)

def todo_add():
    todo = input("What TODO do you want to add: ")
    todos.append(todo)

def todo_remove():
    idx = int(input("Which TODO to remove (number): ")) - 1
    if idx < len(todos):
        print(f"Removing {todos[idx]}")
        todos.pop(idx)
    else:
        print("There is n o TODO at this position")

def bye():
    print("Bye")
    sys.exit(0)

mappings = {
    'stocks': stock_function,
    'todoshow': todo_show,
    'todoadd': todo_add,
    'todoremove': todo_remove,
    'goodbye': bye
}
assistant = BasicAssistant(intents_data="intents.json", method_mappings=mappings)
assistant.fit_model()
assistant.save_model()

while True:
    message = input("Message: ")
    response = assistant.process_input(message)
    print(response)