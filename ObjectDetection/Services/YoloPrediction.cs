using ObjectDetection.Helper;
using ObjectDetection.Models;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;


namespace ObjectDetection.Services
{
    class YoloPrediction
    {

        public static IList<YoloBoundingBox> ObtainBoundingBoxes(string base64Image)
        {

            var opencvImage = Image.ConvertFromBase64ToMat(base64Image);
            MatOfByte3 mat3 = new MatOfByte3(opencvImage);
            var indexer = mat3.GetIndexer();
            Tensor<float> imageData = new DenseTensor<float>(new[] { 3, opencvImage.Width, opencvImage.Height });
            for (int y = 0; y < opencvImage.Height; y++)
            {
                for (int x = 0; x < opencvImage.Width; x++)
                {
                    Vec3b color = indexer[y, x];
                    imageData[0, y, x] = (float)color.Item2;
                    imageData[1, y, x] = (float)color.Item1;
                    imageData[2, y, x] = (float)color.Item0;
                }
            }

            var yoloParser = new YoloParser();
            var resultTransform = imageData.Reshape(new ReadOnlySpan<int>(new[] { 3 * 416 * 416 }));
            var yoloModel = new YoloModel();
            var results = yoloModel.Evaluate(new[] { resultTransform });
            var boundingBoxes = yoloParser.ParseOutputs(results.First().ToArray());

            return boundingBoxes;
        }
    }
}
