import pandas as pd


if __name__ == '__main__':
    df = pd.DataFrame({'A': [1, 2, -1, 0], 'B': [4, 5, 6, -2]})

    print(df)

    # Clip values below 2 in column 'A'
    df = df[df['A'] > 0]

    print(df)

    # Clip values below 0 in the entire DataFrame
    df = df.clip(lower=0)

    print(df)

