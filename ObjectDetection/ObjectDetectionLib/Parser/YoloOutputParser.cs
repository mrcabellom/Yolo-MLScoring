using ObjectDetectionLib.Enums;
using ObjectDetectionLib.Helper;
using ObjectDetectionLib.Models;
using System;
using System.Collections.Generic;
using System.Linq;


namespace ObjectDetectionLib.Parser
{
    public class YoloOutputParser
    {
        public const int RowCount = 13;
        public const int ColCount = 13;
        public const int ChannelCount = 125;
        public const int BoxesPerCell = 5;
        public const int BoxInfoCount = 5;
        public const int ClassCount = 20;
        public const float CellWidth = 32;
        public const float CellHeight = 32;

        private readonly int channelStride = RowCount * ColCount;

        private readonly float[] anchors = new float[]
            {
                1.08F, 1.19F, 3.42F, 4.41F, 6.63F, 11.38F, 9.42F, 5.11F, 16.62F, 10.52F
            };

        public IList<BoundingBox> ParseOutputs(float[] yoloModelOutputs, float threshold = .2F)
        {

            var boxes = new List<BoundingBox>();

            for (int cy = 0; cy < RowCount; cy++)
            {
                for (int cx = 0; cx < ColCount; cx++)
                {
                    for (int b = 0; b < BoxesPerCell; b++)
                    {
                        var channel = (b * (ClassCount + BoxInfoCount));

                        var tx = yoloModelOutputs[GetOffset(cx, cy, channel)];
                        var ty = yoloModelOutputs[GetOffset(cx, cy, channel + 1)];
                        var tw = yoloModelOutputs[GetOffset(cx, cy, channel + 2)];
                        var th = yoloModelOutputs[GetOffset(cx, cy, channel + 3)];
                        var tc = yoloModelOutputs[GetOffset(cx, cy, channel + 4)];

                        var x = (cx + MathMethods.Sigmoid(tx)) * CellWidth;
                        var y = (cy + MathMethods.Sigmoid(ty)) * CellHeight;
                        var width = (float)Math.Exp(tw) * CellWidth * anchors[b * 2];
                        var height = (float)Math.Exp(th) * CellHeight * anchors[b * 2 + 1];

                        var confidence = MathMethods.Sigmoid(tc);
                        if (confidence < threshold)
                        {
                            continue;
                        }

                        var classes = new float[ClassCount];
                        var classOffset = channel + BoxInfoCount;

                        for (int i = 0; i < ClassCount; i++)
                        {
                            classes[i] = yoloModelOutputs[GetOffset(cx, cy, i + classOffset)];
                        }

                        var results = MathMethods.Softmax(classes)
                            .Select((v, iexp) => new { Value = v, Index = iexp });

                        var labelClass = results.OrderByDescending(r => r.Value).First().Index;
                        var scoreClass = results.OrderByDescending(r => r.Value).First().Value * confidence;
                        var testSum = results.Sum(r => r.Value);

                        if (scoreClass > threshold)
                        {
                            boxes.Add(new BoundingBox()
                            {
                                Confidence = scoreClass,
                                X = (x - width / 2),
                                Y = (y - height / 2),
                                Width = width,
                                Height = height,
                                Label = Enum.GetName(typeof(ObjectLabels), labelClass).ToLower()
                            });
                        }
                    }
                }
            }

            var filteredBoxes = NonMaxSuppression(boxes, 5, .5F);
            return filteredBoxes;
        }

        public IList<BoundingBox> NonMaxSuppression(IList<BoundingBox> boxes, int limit, float threshold)
        {
            var numberActiveBoxes = boxes.Count;
            var isActiveBoxes = new bool[numberActiveBoxes];

            for (int i = 0; i < isActiveBoxes.Length; i++)
            {
                isActiveBoxes[i] = true;
            }

            var sortedBoxes = boxes.Select((b, i) => new { Box = b, Index = i })
                                .OrderByDescending(b => b.Box.Confidence)
                                .ToList();

            var results = new List<BoundingBox>();
            for (int i = 0; i < boxes.Count; i++)
            {
                if (isActiveBoxes[i])
                {
                    var boxA = sortedBoxes[i].Box;
                    results.Add(boxA);
                    if (results.Count >= limit)
                    {
                        break;
                    }

                    for (var j = i + 1; j < boxes.Count; j++)
                    {
                        if (isActiveBoxes[j])
                        {
                            var boxB = sortedBoxes[j].Box;
                            if (IntersectionOverUnion(boxA, boxB) > threshold)
                            {
                                isActiveBoxes[j] = false;
                                numberActiveBoxes--;
                                if (numberActiveBoxes <= 0)
                                {
                                    break;
                                }
                            }
                        }
                    }
                    if (numberActiveBoxes <= 0)
                    {
                        break;
                    }
                }
            }
            return results;
        }

        private float IntersectionOverUnion(BoundingBox a, BoundingBox b)
        {
            var areaA = a.Width * a.Height;
            var areaB = b.Width * b.Height;
            if (areaA <= 0 || areaB <= 0)
            {
                return 0;
            }

            var minXRectangle = Math.Max(a.X, b.X);
            var minYRectangle = Math.Max(a.Y, b.Y);
            var maxXRectangle = Math.Min(a.X + a.Width, b.X + b.Width);
            var maxYRectangle = Math.Min(a.Y + a.Height, b.Y + b.Height);

            var intersectionArea = Math.Max(maxYRectangle - minYRectangle, 0) * Math.Max(maxXRectangle - minXRectangle, 0);
            return intersectionArea / (areaA + areaB - intersectionArea);
        }

        private int GetOffset(int x, int y, int channel)
        {
            return (channel * channelStride) + (y * ColCount) + x;
        }

    }
}
