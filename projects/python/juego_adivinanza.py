#************* Crear un nuego de adivinanza *******************************

# El programa va a preguntar el usuario de su nombre
# Le va a decir quue tiene 8 intentos para adivinar el número. 
# En cada intento, el jugador dirá un número y el programa puede responder de las siguentes maneras:
#    si el número del usuario es < 1 o > 100, el programa dice que el número no está permitido
#    si el número es menor al que ha pensado el programa, le va a decir que el número es menor al número secreto
#    si el usuario eligió un número mayor, el programa le dice que el número es mayor e incorrecto
#    si el usuario acertó el número secreto, se le informa que ha ganado y cuantos intentos le ha tomado

import random

nombre_usuario = input('Dime tu nombre ')
print(f'Hola {nombre_usuario}. He pensado en un número entre 1 y 100. Tienes solo 8 intentos para adivinar')

numero_programa = random.randint(1,100)

vidas = 8

while vidas > 0:
    numero_usuario = int(input('Cuál crees que es el número '))

    if numero_usuario < 1 or numero_usuario > 100:
        print('Número fuera del rango permitido.')
        continue

    if numero_usuario < numero_programa:
        print('El numero secreto es mayor.')
        vidas -= 1

    elif numero_usuario > numero_programa:
        print('El numero secreto es menor.')
        vidas -= 1

    else:
        print('Has ganado!!! Enhorabuena!')
        print(f'Te han costado {9 - vidas} intentos.')
        break

else:
    print(f'Has perdido. El número era {numero_programa}')




