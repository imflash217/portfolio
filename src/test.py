# Sample dummy Python script
def calculate_fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]

    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

def main():
    # Example usage
    n = 10
    result = calculate_fibonacci(n)
    print(f"First {n} numbers in Fibonacci sequence: {result}")

    # Simple list manipulation
    numbers = [1, 2, 3, 4, 5]
    squared = [x**2 for x in numbers]
    print(f"Original numbers: {numbers}")
    print(f"Squared numbers: {squared}")

if __name__ == "__main__":
    main()
