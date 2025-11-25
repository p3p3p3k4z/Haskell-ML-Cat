{-# LANGUAGE OverloadedStrings #-}

module Main where

import Codec.Picture
import Codec.Picture.Types
import qualified Data.Vector.Storable as V
import qualified Data.Vector as VBoxed

import qualified Data.ByteString.Lazy.Char8 as BL
import Data.Csv (encode)
import Data.List (intersperse)
import System.Directory (listDirectory, doesDirectoryExist)
import System.FilePath ((</>), takeFileName)
import Control.Monad (forM, filterM)
import System.Random (randomRIO)
import Data.Word (Word8)

-- Configuración
binsPerChannel :: Int
binsPerChannel = 8
imgLimitPerClass :: Int
imgLimitPerClass = 500
outputCsvFile :: FilePath
outputCsvFile = "training_data.csv"
baseDataDir :: FilePath
baseDataDir = "Cat_Breed"

-- Función para calcular histograma de color 3D
-- Preparar los datos
calculateColorHistogram :: Image PixelRGB8 -> [Double]
calculateColorHistogram img =
    let width = imageWidth img
        height = imageHeight img
        totalPixels = width * height
        -- De matriz de pixeles a vectores de caracteristicas
        -- Inicializar histograma 3D (8x8x8 = 512 bins)
        emptyHistogram = VBoxed.replicate (binsPerChannel * binsPerChannel * binsPerChannel) 0.0
        
        -- Función para calcular el bin para un valor de canal (0-255)
        calculateBin :: Word8 -> Int
        calculateBin val = fromIntegral val `div` 32  -- 256 / 8 = 32
        
        -- Procesar cada píxel y acumular en el histograma
        histogram = foldl' updateHistogram emptyHistogram 
                   [pixelAt img x y | y <- [0..height-1], x <- [0..width-1]]
        
        -- Actualizar el histograma para un píxel
        updateHistogram hist pixel =
            let r = fromIntegral (pixelAtX pixel 0) :: Int
                g = fromIntegral (pixelAtX pixel 1) :: Int
                b = fromIntegral (pixelAtX pixel 2) :: Int
                
                rBin = calculateBin (fromIntegral r)
                gBin = calculateBin (fromIntegral g) 
                bBin = calculateBin (fromIntegral b)
                
                -- Calcular índice 1D en el histograma
                index = (rBin * binsPerChannel * binsPerChannel) + 
                        (gBin * binsPerChannel) + 
                        bBin
                
                current = hist VBoxed.! index
            in hist VBoxed.// [(index, current + 1.0)]
        
        -- Normalizar el histograma (convertir a frecuencias)
        normalizedHist = VBoxed.map (/ fromIntegral totalPixels) histogram
        
    in VBoxed.toList normalizedHist

-- Versión alternativa usando imageData 
calculateColorHistogramEfficient :: Image PixelRGB8 -> [Double]
calculateColorHistogramEfficient img =
    let width = imageWidth img
        height = imageHeight img
        totalPixels = width * height
        
        -- Obtener datos de la imagen como vector
        pixelData = imageData img
        pixelList = V.toList pixelData
        
        -- Inicializar histograma
        emptyHistogram = VBoxed.replicate (binsPerChannel * binsPerChannel * binsPerChannel) 0.0
        
        -- Procesar píxeles en grupos de 3 (R, G, B)
        processPixelGroup :: VBoxed.Vector Double -> [Word8] -> VBoxed.Vector Double
        processPixelGroup hist (r:g:b:rest) =
            let rBin = fromIntegral r `div` 32
                gBin = fromIntegral g `div` 32
                bBin = fromIntegral b `div` 32
                
                index = (rBin * binsPerChannel * binsPerChannel) + 
                        (gBin * binsPerChannel) + 
                        bBin
                
                current = hist VBoxed.! index
                updatedHist = hist VBoxed.// [(index, current + 1.0)]
            in processPixelGroup updatedHist rest
        processPixelGroup hist _ = hist
        
        -- Construir histograma
        histogram = processPixelGroup emptyHistogram pixelList
        
        -- Normalizar
        normalizedHist = VBoxed.map (/ fromIntegral totalPixels) histogram
        
    in VBoxed.toList normalizedHist

processImage :: FilePath -> IO (Maybe [Double])
processImage path = do
    eitherImg <- readImage path
    case eitherImg of
        Left err -> do
            putStrLn $ "Advertencia: No se pudo leer " ++ path ++ ": " ++ err
            return Nothing
        Right dynamicImage -> do
            let img = convertRGB8 dynamicImage
            -- Usar cualquiera de las dos funciones de histograma
            let features = calculateColorHistogramEfficient img
            return (Just features)

-- Lógica de selección aleatoria
selectRandomFiles :: [FilePath] -> Int -> IO [FilePath]
selectRandomFiles files limit
    | length files <= limit = return files
    | otherwise = do
        shuffled <- shuffle files
        return $ take limit shuffled

shuffle :: [a] -> IO [a]
shuffle list = shuffle' list []
  where
    shuffle' [] acc = return acc
    shuffle' lst acc = do
      i <- randomRIO (0, length lst - 1)
      let (before, x:after) = splitAt i lst
      shuffle' (before ++ after) (x:acc)

-- Procesar directorio de clase con lógica de NotCat como en Python
processClassDir :: FilePath -> String -> IO [[(String, [Double])]]
processClassDir dirPath label = do
    putStrLn $ "Procesando carpeta: " ++ label
    
    allFiles <- listDirectory dirPath
    let imageFiles = filter isImageFile allFiles
    
    if label == "NotCat"
        then processNotCatDir dirPath imageFiles
        else processCatBreedDir dirPath label imageFiles

-- Procesar razas de gato (muestreo aleatorio simple)
processCatBreedDir :: FilePath -> String -> [FilePath] -> IO [[(String, [Double])]]
processCatBreedDir dirPath label imageFiles = do
    let allPaths = map (dirPath </>) imageFiles
    selectedPaths <- selectRandomFiles allPaths imgLimitPerClass
    putStrLn $ "  ... " ++ show (length selectedPaths) ++ " imágenes seleccionadas (de " ++ show (length imageFiles) ++ ")."
    
    processImageBatch label selectedPaths

-- Procesar NotCat con muestreo estratificado (como en Python)
processNotCatDir :: FilePath -> [FilePath] -> IO [[(String, [Double])]]
processNotCatDir dirPath imageFiles = do
    putStrLn "  Procesando NotCat con muestreo estratificado..."
    
    -- Obtener subdirectorios de NotCat
    subdirs <- filterM doesDirectoryExist (map (dirPath </>) imageFiles)
    
    allResults <- forM subdirs $ \subdirPath -> do
        let subclassName = takeFileName subdirPath
        putStrLn $ "    Procesando sub-clase: " ++ subclassName
        
        subclassFiles <- listDirectory subdirPath
        let subclassImageFiles = filter isImageFile subclassFiles
        let subclassPaths = map (subdirPath </>) subclassImageFiles
        
        -- Seleccionar máximo 200 por subclase (como en Python)
        selectedSubclassPaths <- selectRandomFiles subclassPaths 200
        putStrLn $ "      ... " ++ show (length selectedSubclassPaths) ++ " imágenes seleccionadas (de " ++ show (length subclassImageFiles) ++ ")."
        
        processImageBatch "NotCat" selectedSubclassPaths
    
    return (concat allResults)

-- Procesar un lote de imágenes
processImageBatch :: String -> [FilePath] -> IO [[(String, [Double])]]
processImageBatch label paths = do
    results <- forM paths $ \path -> do
        maybeFeatures <- processImage path
        case maybeFeatures of
            Nothing -> return []
            Just features -> return [(label, features)]
    return results

-- Detección de archivos de imagen
isImageFile :: FilePath -> Bool
isImageFile path = 
    let lowerPath = map toLower path
    in any (`isSuffixOf` lowerPath) [".jpg", ".jpeg", ".png", ".bmp"]

toLower :: Char -> Char
toLower c
    | c >= 'A' && c <= 'Z' = toEnum (fromEnum c + 32)
    | otherwise = c

isSuffixOf :: String -> String -> Bool
isSuffixOf suffix str = 
    suffix == drop (length str - length suffix) str

-- Función auxiliar foldl
foldl' :: (b -> a -> b) -> b -> [a] -> b
foldl' f z [] = z
foldl' f z (x:xs) = let z' = f z x in z' `seq` foldl' f z' xs

main :: IO ()
main = do
    putStrLn "--- Iniciando Extractor de Características con Histogramas ---"
    putStrLn "¡Modo de Muestreo Estratificado!"
    putStrLn "  - Razas de Gato: Máx 500 por raza."
    putStrLn "  - Clase 'NotCat': Máx 200 por sub-clase."

    -- Encontrar carpetas de clases
    classDirs <- listDirectory baseDataDir
    let classPaths = map (baseDataDir </>) classDirs
    dirPathsOnly <- filterM doesDirectoryExist classPaths
    let labels = map takeFileName dirPathsOnly

    putStrLn $ "Encontradas " ++ show (length labels) ++ " clases: " ++ show labels

    -- Procesar cada carpeta
    results <- mapM (uncurry processClassDir) (zip dirPathsOnly labels)
    let flatResults = concat $ concat results
    
    -- Crear CSV
    let featureCount = binsPerChannel * binsPerChannel * binsPerChannel
    let header = "label" : map (("f"++) . show) [1..featureCount]

    putStrLn $ "\nEscribiendo archivo " ++ outputCsvFile ++ " con " ++ show (length flatResults) ++ " muestras..."
    
    let csvContent = unlines $ 
            [BL.unpack $ BL.init $ encode [header]] ++
            map (\(label, features) -> 
                label ++ "," ++ (concat $ intersperse "," $ map show features)
            ) flatResults
    
    BL.writeFile outputCsvFile (BL.pack csvContent)
    putStrLn "¡Extracción completada!"