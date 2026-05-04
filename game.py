import random


def start_game():
    return [
        ["b", "b", "b"],
        ["b", "b", "b"],
        ["b", "b", "b"],
    ]


def print_board(board):
    for row in board:
        print(" ".join(cell if cell != "b" else "." for cell in row))
    print()


def get_empty_positions(board):
    empty = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == "b":
                empty.append((i, j))
    return empty


def check_winner(board):
    # Linhas e colunas
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != "b":
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != "b":
            return board[0][i]

    # Diagonais
    if board[0][0] == board[1][1] == board[2][2] != "b":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != "b":
        return board[0][2]

    return None


def is_draw(board):
    return all(cell != "b" for row in board for cell in row)


def player_move(board):
    while True:
        try:
            move = input("Digite sua jogada (linha coluna, de 0 a 2): ")
            i, j = map(int, move.split())

            if i not in range(3) or j not in range(3):
                print("Posição inválida!")
                continue

            if board[i][j] != "b":
                print("Posição já ocupada!")
                continue

            board[i][j] = "x"
            break
        except:
            print("Entrada inválida! Use: linha coluna")


def random_ai_move(board):
    empty = get_empty_positions(board)
    if empty:
        i, j = random.choice(empty)
        board[i][j] = "o"
        print(f"IA jogou em: {i} {j}")


def play():
    board = start_game()
    print("Você é X. A IA é O.\n")

    while True:
        print_board(board)

        # Jogador (X)
        player_move(board)
        winner = check_winner(board)
        if winner:
            print_board(board)
            print("Você venceu!")
            break
        if is_draw(board):
            print_board(board)
            print("Empate!")
            break

        # IA (O)
        random_ai_move(board)
        winner = check_winner(board)
        if winner:
            print_board(board)
            print("IA venceu!")
            break
        if is_draw(board):
            print_board(board)
            print("Empate!")
            break


if __name__ == "__main__":
    play()