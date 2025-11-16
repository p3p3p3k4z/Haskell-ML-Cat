{-# LANGUAGE OverloadedStrings #-}

module Main where

import Codec.Picture
import Codec.Picture.Types
import qualified Data.Vector.Storable as V

import qualified Data.ByteString.Lazy.Char8 as BL
import Data.Csv (encode)
import Data.List (intersperse)
import System.Directory (listDirectory, doesDirectoryExist)
import System.FilePath ((</>), takeFileName)
import Control.Monad (forM, filterM)
import System.Random (randomRIO)
import Data.Word (Word8)

-- Configuración
imgWidth, imgHeight :: Int
imgWidth = 32
imgHeight = 32
imgLimitPerClass :: Int
imgLimitPerClass = 500
outputCsvFile :: FilePath
outputCsvFile = "training_data.csv"
baseDataDir :: FilePath
baseDataDir = "Cat_Breed"

-- Función simple de redimensionamiento
resizeImage :: Int -> Int -> Image PixelRGB8 -> Image PixelRGB8
resizeImage newWidth newHeight img = 
    generateImage getPixel newWidth newHeight
  where
    oldWidth = imageWidth img
    oldHeight = imageHeight img
    xRatio = fromIntegral oldWidth / fromIntegral newWidth
    yRatio = fromIntegral oldHeight / fromIntegral newHeight
    
    getPixel x y =
        let oldX = floor (fromIntegral x * xRatio)
            oldY = floor (fromIntegral y * yRatio)
            oldX' = min oldX (oldWidth - 1)
            oldY' = min oldY (oldHeight - 1)
        in pixelAt img oldX' oldY'

-- Extrae características RGB usando el vector de datos de la imagen
imageToFeatures :: Image PixelRGB8 -> [Double]
imageToFeatures img =
    let resizedImg = resizeImage imgWidth imgHeight img
        -- Obtener el vector de datos de la imagen
        pixelVector = imageData resizedImg
        -- Convertir el vector a lista y normalizar
        features = map (\w -> fromIntegral (w :: Word8) / 255.0) (V.toList pixelVector)
    in features

processImage :: FilePath -> IO (Maybe [Double])
processImage path = do
    eitherImg <- readImage path
    case eitherImg of
        Left err -> do
            putStrLn $ "Advertencia: No se pudo leer " ++ path ++ ": " ++ err
            return Nothing
        Right dynamicImage -> do
            let img = convertRGB8 dynamicImage
            let features = imageToFeatures img
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

processClassDir :: FilePath -> String -> IO [[(String, [Double])]]
processClassDir dirPath label = do
    putStrLn $ "Procesando carpeta: " ++ label
    allFiles <- listDirectory dirPath
    let imageFiles = filter isImageFile allFiles
    let allPaths = map (dirPath </>) imageFiles
    
    selectedPaths <- selectRandomFiles allPaths imgLimitPerClass
    putStrLn $ "  ... " ++ show (length selectedPaths) ++ " imágenes seleccionadas (de " ++ show (length allFiles) ++ ")."

    results <- forM selectedPaths $ \path -> do
        maybeFeatures <- processImage path
        case maybeFeatures of
            Nothing -> return []
            Just features -> return [(label, features)]
    
    return results

-- Detección de archivos de imagen mejorada
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

-- Main
main :: IO ()
main = do
    putStrLn "--- Iniciando Extractor de Características ---"

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
    let featureCount = imgWidth * imgHeight * 3
    let header = "label" : map (("f"++) . show) [1..featureCount]

    putStrLn $ "\nEscribiendo archivo " ++ outputCsvFile ++ " con " ++ show (length flatResults) ++ " muestras..."
    
    let csvContent = unlines $ 
            [BL.unpack $ BL.init $ encode [header]] ++
            map (\(label, features) -> 
                label ++ "," ++ (concat $ intersperse "," $ map show features)
            ) flatResults
    
    BL.writeFile outputCsvFile (BL.pack csvContent)
    putStrLn "¡Extracción completada!"