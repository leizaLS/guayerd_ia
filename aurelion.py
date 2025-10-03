import os, platform, subprocess, sys, pandas as pd
from textos import document

#dependencies
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install("openpyxl")

def cleanConsole():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def showMenu():
    print("**MENÚ PRINCIPAL**"  +
    "\n1. Tema, problema y solución" + 
    "\n2. Dataset de referencia" +
    "\n3. Estructura por tabla (tipo y escala)" +
    "\n4. Escalas de medición" +
    "\n5. Sugerencias y mejoras con Copilot" +
    "\n6. Salir" )

def showData(fileName):
    dir = os.path.dirname(os.path.abspath(__file__))

    # buscar archivo en carpeta db/
    root = os.path.join(dir,"db", fileName)  
    try:
        df = pd.read_excel(root)
        print(df.head()) 
    except Exception as e:
        print(f"Error al leer {fileName}: {e}")

def menuData():
    while True:
        cleanConsole()
        print("\n** MENÚ DE VISUALIZACIÓN DE DATOS **" +
              "\n1. Clientes" +
              "\n2. Detalle de ventas" +
              "\n3. Productos" +
              "\n4. Ventas" +
              "\n5. Volver al menú principal")

        option = input("\nSelecciona una opción (1-5): ").strip()

        if option == "1":
            showData("clientes.xlsx")
        elif option == "2":
            showData("detalle_ventas.xlsx")
        elif option == "3":
            showData("productos.xlsx")
        elif option == "4":
            showData("ventas.xlsx")
        elif option == "5":
            break
        else:
            print("Opción no válida. Intenta de nuevo.")

        input("\nPresiona Enter para continuar...")

def main():
    while True:
        cleanConsole()
        showMenu()
        option = input("\nSelecciona una opción (1-6): ").strip()

        if option in document:
            cleanConsole()
            print(document[option])

            if option == "2":
                menuData()
            else:
                input("\n*Presione Enter para volver al menú principal...")
        elif option == "6":
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")
            input("\nPresiona Enter para continuar...")

#main
main()