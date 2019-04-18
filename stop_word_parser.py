"""
    Parse stop words into a single file
"""
import json


if __name__ == "__main__":

    stop_word = []

    with open('./Data/stop_word1.txt', 'r') as file:
        for line in file:
            if line not in stop_word:
                stop_word.append(line)

    with open('./Data/stop_word2.txt', 'r') as file:
        for line in file:
            if line not in stop_word:
                stop_word.append(line)
    
    with open('./Data/stop_word.txt', 'w') as file:
        file.writelines(stop_word)