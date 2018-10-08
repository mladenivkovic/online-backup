#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 18:28:24 2017

@author: mivkov
"""

#----------------
#Set up
#----------------


# Read in secret word and allowed number of guesses
secret_word = input("type in the secret word: ")
nguess = input("how many tries are allowed? ")

#make a number out of nguess
nguess = int(nguess)

print("The length of the secret word is:", len(secret_word))







found_word = False                      # is word found?
ntries = 0                              # number of tries
guessed_letters_not_in_word = []        # wrongly guessed letters
guessed_letters_in_word = ['_']*len(secret_word) # correctly guessed letters



#-------------
# Game loop
#-------------

while (ntries < nguess): #aborts as soon as one condition is false7
    
    #------------------------------
    # Write informations
    #------------------------------
    
    print("Current state:", end=' ') #don't make a new line after printing this
    
    for char in guessed_letters_in_word:
        print(char, end=' ') #don't make a new line every time you print
    print() # make a newline after all currently guessed letters or dashes are printed
    
    print("Guessed letters not in the secret word:", end=' ')
    for char in guessed_letters_not_in_word:
        print(char, end=', ') #don't make a new line every time you print
    print() # make a newline after all currently wrongly guessed letters
  
    
    print("Number of tries remaining:", nguess-ntries)
    
    
    
    #----------------------------
    # Guessing game
    #----------------------------
    
    # repeat guessing until you put in a character you haven't put in before
    character_already_guessed = True
    while character_already_guessed:    
        guessed_letter = input("guess a letter: ")
        
        # if letter has been guessed already:
        if (guessed_letter in guessed_letters_in_word) or (guessed_letter in guessed_letters_not_in_word):
            print("You have tried the letter", guessed_letter, "before. Try something else.")
        else:
            # Don't repeat the guess
            character_already_guessed = False
    
    
    
    letter_is_not_in_word = True
    
    for ind, char in enumerate(secret_word): # loop over every character of secret word
        if guessed_letter == char:
            guessed_letters_in_word[ind] = char
            letter_is_not_in_word = False    # if it is in the word at least once, it will be
                                             # set to False and never reset to True for this letter
    
    
    # Check whether the complete word has been found:
    # if it has, there are no more dashes in the 'guessed_letters_in_word' list
    
    if '_' not in guessed_letters_in_word:
        found_word = True
        break # This stops the game/ the while loop.
        
    
    
    # If the letter is not in the word, append it to the list of
    # wrongly guessed letters
    if letter_is_not_in_word:
        guessed_letters_not_in_word.append(guessed_letter)
    else:
        # If it was in the word, don't count this as a try, restart game loop
        continue
        
        
    
    #-------------------------
    # Finishing touches
    #-------------------------
    

    # raise the counter of how many tries you had
    ntries += 1
    
    
    
#==========================================================
# Game loop ends here
#==========================================================    
    
    
if found_word: #word has been found. Congratulate!
    print("You found the word! Congratulations!")
else:
    print("Didn't win this time. Try again!")
        
        
        