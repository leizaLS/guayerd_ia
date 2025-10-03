import os
import platform
from textos import documentacion

def cleanConsole():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def showMenu():
    print("**MENÚ**"  +
    "\n1. Tema, problema y solución" + 
    "\n2. Dataset de referencia" +
    "\n3. Estructura por tabla (tipo y escala)" +
    "\n4. Escalas de medición" +
    "\n5. Sugerencias y mejoras con Copilot" +
    "\n6. Salir" )

def main():
    while True:
        cleanConsole()
        showMenu()
        option = input("\nSelecciona una opción (1-6): ").strip()

        if option in documentacion:
            cleanConsole()
            print(documentacion[option])

            #option 2. Display data from xlsx
            if (option == "2"):
                print("Mostrar menu de datos db")
                #.
                #.
                #.
                #.
            else:
                #return to menu
                input("\n*Presione Enter para volver al menú principal...")
        elif option == "6":
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")
            input("\nPresiona Enter para continuar...")

main()