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

-- Un 'String' es más flexible para las 15+ razas para CSV
-- data Label = Persian | Egyptian | Common | NotCat deriving (Show, Eq)
type Label = String

-- FeatureVector es simplemente una lista de números
-- Representando el histograma, espacio
type FeatureVector = [Double]

-- LabeledPoint es el "Tipo Producto" que une una Etiqueta
-- con su Vector de Características (ejemplos de entrenamiento)
data LabeledPoint = LabeledPoint {
    lpLabel    :: Label,
    lpFeatures :: FeatureVector
} deriving (Show, Eq)


-- 2. FUNCIONES PURAS
{- |
Calcula la Distancia Euclidiana entre dos vectores (listas de Dobles).
Esta es la fórmula implementada: d(p,q) = sqrt( sum( (pi - qi)^2 ) )
Realmente mide que tan similares son las imagenes en color
-}
euclideanDistance :: FeatureVector -> FeatureVector -> Double
-- Restar histograma (p1-q1..)
euclideanDistance p q = sqrt $ sum $ zipWith (\x y -> (x - y)^2) p q

{- |
Encuentra la etiqueta más común en una lista de etiquetas.
Esta es la función de "votación multi-clase".
Si los 5 vecinos son ["Persa", "Persa", "Siamés", "Persa", "Siamés"],
esta función debe devolver "Persa" porque tiene 3 votos.
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
    -- 4. Toma el primer elemento del grupo más largo (ej. "A") y extrae la etiqueta
    in head $ head longestGroupFirst

{- |
Función principal del clasificador KNN.
Encuentra la etiqueta predicha para un nuevo vector de características.
Aqui es donde estra el aprendizaje supervisado, pero lo autonombre aprendijaze perezoso
Ya que en realidad no entrena un modelo, memoriza los datos y busca en el momento
-}
kNearestNeighbors :: Int -> [LabeledPoint] -> FeatureVector -> Label
kNearestNeighbors k trainingData query =
    let
        -- Medir Similitud (Fuerza Bruta)
        -- Calculamos la distancia entre la imagen nueva ('query') 
        -- y TODAS las imágenes de la base de datos.
        -- Resultado: Una lista de pares [(Distancia, "Raza"), ...]
        distances = map (\point -> (euclideanDistance (lpFeatures point) query, lpLabel point)) trainingData
        
        -- Ranking
        -- Ordenamos la lista de menor a mayor distancia.
        -- Los que están arriba son los "Vecinos Más Cercanos".
        sortedDistances = sortBy (comparing fst) distances
        
        -- Toma las primeras 'k' tuplas (los 'k' vecinos más cercanos)
        kNearest = take k sortedDistances
        
        -- Extrae solo las etiquetas de esos 'k' vecinos
        -- Descartamos las distancias numéricas, solo nos importan las etiquetas.
        -- Ej: [(0.1, "Persa"), (0.2, "Siamés")] -> ["Persa", "Siamés"]
        kNearestLabels = map snd kNearest
    in
        -- Llama a 'majorityVote' para encontrar la etiqueta ganadora
        -- Contamos los votos para ver quién gana.
        majorityVote kNearestLabels