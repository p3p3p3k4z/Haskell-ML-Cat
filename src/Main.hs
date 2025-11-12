-- Este módulo SÍ necesita dependencias externas
-- Necesitarás la biblioteca 'split' para 'splitOn'
import Data.List.Split (splitOn)
import System.Environment (getArgs)
import System.Exit (die)
import Control.Monad (liftM)

-- ¡Importamos nuestro propio módulo de lógica pura!
import Classifier

-- 1. FUNCIONES DE PARSEO (Ayudantes)
-- ------------------------------------------------

{- |
Convierte una línea del CSV en un LabeledPoint.
Ej. "Persian,0.1,0.2" -> LabeledPoint "Persian" [0.1, 0.2]
-}
parseLine :: String -> LabeledPoint
parseLine line =
    let parts = splitOn "," line  -- Separa la línea por comas
        label = head parts        -- La etiqueta es la primera parte
        -- 'tail parts' son todas las demás partes
        -- 'map read' convierte cada "String" de característica en un "Double"
        features = map read (tail parts)
    in LabeledPoint label features

{- |
Convierte el argumento de la CLI en un FeatureVector.
Ej. "0.1,0.2" -> [0.1, 0.2]
-}
parseQuery :: String -> FeatureVector
parseQuery queryString = map read (splitOn "," queryString)

-- 2. LA FUNCIÓN PRINCIPAL (El mundo "Impuro" de IO)
-- ------------------------------------------------

main :: IO ()
main = do
    --
    putStrLn "--- Clasificador Funcional de Gatos (Haskell) ---"
    
    -- --- Tarea de IO: Cargar el set de entrenamiento ---
    -- 'readFile' es una acción de IO que lee el archivo.
    -- 'csvContent' es un String que contiene TODO el archivo.
    putStrLn "Cargando 'training_data.csv'..."
    csvContent <- readFile "training_data.csv"
    
    -- --- Tarea Pura: Parsear los datos ---
    -- 'lines' divide el String en una lista de Strings (líneas)
    -- 'map parseLine' aplica nuestra función pura a cada línea.
    -- 'trainingData' es ahora nuestra lista pura: [LabeledPoint]
    let trainingData = map parseLine (lines csvContent)
    putStrLn $ "¡Carga completa! " ++ (show $ length trainingData) ++ " puntos de datos listos."

    -- --- Tarea de IO: Leer argumentos de la CLI ---
    args <- getArgs
    
    -- Validar que tengamos el argumento
    queryVector <- case args of
        -- Si 'args' es una lista con un elemento (nuestro vector)
        [queryStr] -> return (parseQuery queryStr)
        -- Si no, morimos y mostramos un error
        _          -> die "Error: Por favor, proporciona el vector de características como argumento. \nEjemplo: ./fun_cat_classifier 0.1,0.2,..."
    
    -- --- Tarea Pura: Ejecutar el algoritmo KNN ---
    --
    
    -- Definimos nuestro valor K
    let k = 5
    
    -- ¡Llamamos a nuestra función 100% pura!
    let prediction = kNearestNeighbors k trainingData queryVector
    
    -- --- Tarea de IO: Imprimir el resultado a stdout ---
    --
    -- Esta es la única línea que Python leerá
    putStrLn prediction