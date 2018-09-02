using ObjectDetectionLib.Helper;
using ObjectDetectionLib.Models;
using ObjectDetectionLib.Parser;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using YoloInference;

namespace ObjectDetectionLib.Services
{
    public class YoloPrediction
    {

        public const int channels = 3;

        public static IList<BoundingBox> ObtainBoundingBoxes(string base64Image)
        {
            var opencvImage = Image.ConvertFromBase64ToMat(base64Image);
            var tensor = new DenseTensor<float>(new int[] { channels, opencvImage.Height, opencvImage.Width });

            using (var mat = new MatOfByte3(opencvImage))
            {
                var indexer = mat.GetIndexer();
                for (int y = 0; y < opencvImage.Height; y++)
                {
                    for (int x = 0; x < opencvImage.Width; x++)
                    {
                        Vec3b color = indexer[y, x];
                        tensor[0, y, x] = (float)color.Item2;
                        tensor[1, y, x] = (float)color.Item1;
                        tensor[2, y, x] = (float)color.Item0;
                    }
                }
            }

            var transform = tensor.Reshape(new ReadOnlySpan<int>(new[] { channels * opencvImage.Height * opencvImage.Width }));
            var yoloParser = new YoloOutputParser();
            var yoloModel = YoloModel.Instance;
            var results = yoloModel.Evaluate(new[] { transform });
            return yoloParser.ParseOutputs(results.First().ToArray());
        }
    }
}
