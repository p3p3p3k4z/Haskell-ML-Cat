-- Declaramos este archivo como un módulo reutilizable
module Classifier (
    Label,
    FeatureVector,
    LabeledPoint(..), -- Exportamos el tipo y sus constructores
    euclideanDistance,
    kNearestNeighbors,
    majorityVote
) where

-- Importamos funciones útiles para listas
import Data.List (sortBy, groupBy, sort)
import Data.Ord (comparing)

-- 1. DEFINICIONES DE TIPOS (ADTs)
--
-- ------------------------------------------------

-- Un 'String' es más flexible para las 15+ razas de tu CSV
type Label = String

-- Un FeatureVector es simplemente una lista de números (Dobles)
type FeatureVector = [Double]

-- Un LabeledPoint es el "Tipo Producto" que une una Etiqueta
-- con su Vector de Características.
data LabeledPoint = LabeledPoint {
    lpLabel    :: Label,
    lpFeatures :: FeatureVector
} deriving (Show, Eq)


-- 2. FUNCIONES PURAS
--
-- ------------------------------------------------

{- |
Calcula la Distancia Euclidiana entre dos vectores (listas de Dobles).
Esta es la fórmula implementada: d(p,q) = sqrt( sum( (pi - qi)^2 ) )

-}
euclideanDistance :: FeatureVector -> FeatureVector -> Double
euclideanDistance p q = sqrt $ sum $ zipWith (\x y -> (x - y)^2) p q

{- |
Encuentra la etiqueta más común en una lista de etiquetas.
Esta es la función de "votación multi-clase".
-}
majorityVote :: [Label] -> Label
majorityVote labels =
    -- 1. Ordena las etiquetas (ej. ["A", "B", "A"] -> ["A", "A", "B"])
    let sortedLabels = sort labels
    -- 2. Agrupa (ej. [["A", "A"], ["B"]])
        groupedLabels = groupBy (==) sortedLabels
    -- 3. Ordena los grupos por su *longitud* (de mayor a menor)
        -- sortBy 'comparing length' ordena de menor a mayor
        -- (reverse) lo invierte para tener el más largo primero
        longestGroupFirst = reverse $ sortBy (comparing length) groupedLabels
    -- 4. Toma el primer elemento del grupo más largo (ej. "A")
    in head $ head longestGroupFirst

{- |
Función principal del clasificador KNN.
Encuentra la etiqueta predicha para un nuevo vector de características.

-}
kNearestNeighbors :: Int -> [LabeledPoint] -> FeatureVector -> Label
kNearestNeighbors k trainingData query =
    let
        -- 1. Calcula la distancia desde 'query' a CADA punto en 'trainingData'
        distances = map (\point -> (euclideanDistance (lpFeatures point) query, lpLabel point)) trainingData
        
        -- 2. Ordena la lista por la distancia (el primer elemento de la tupla)
        sortedDistances = sortBy (comparing fst) distances
        
        -- 3. Toma las primeras 'k' tuplas (los 'k' vecinos más cercanos)
        kNearest = take k sortedDistances
        
        -- 4. Extrae solo las etiquetas de esos 'k' vecinos
        kNearestLabels = map snd kNearest
    in
        -- 5. Llama a 'majorityVote' para encontrar la etiqueta ganadora
        majorityVote kNearestLabels