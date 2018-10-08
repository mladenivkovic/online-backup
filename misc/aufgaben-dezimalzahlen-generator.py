#!/usr/bin/python

import random as r

def get_decimal():
    nachkommastellen=r.randint(1,6)
    vorkommastellen=r.randint(1,3)
    dec = round(r.uniform(10.0**(vorkommastellen-1), 10.0**vorkommastellen), nachkommastellen)
    return dec


def get_addition():
    print ""
    print ""
    anzahl=raw_input('Aufgaben Addition\n')
    print ""
    anzahl=int(anzahl)
    for i in range(1, anzahl+1):
        a = get_decimal()
        b = get_decimal()
        c = get_decimal()
        d = get_decimal()
        print('{0:4}{1:4}{2:10}{3:4}{4:10}{5:6}{6:4}{7:10}{8:4}{9:10}'.format(str(i)+')','a)',str(a),' + ',str(b),' ','b)',str(c),' + ',str(d) ))
    print ""

def get_subtraktion():
    print ""
    print ""
    anzahl=raw_input('Aufgaben Subtraktion\n')
    print ""
    anzahl=int(anzahl)
    for i in range(1, anzahl+1):
        a = get_decimal()
        b = a + 1
        while (b > a) :
            b = get_decimal()

        c = get_decimal()
        d = c + 1
        while (d > c) :
            d = get_decimal()
        
        print('{0:4}{1:4}{2:10}{3:4}{4:10}{5:6}{6:4}{7:10}{8:4}{9:10}'.format(str(i)+')','a)',str(a),' - ',str(b),' ','b)',str(c),' - ',str(d) ))
    print ""


if __name__ == "__main__" :
    get_addition()
    get_subtraktion()
