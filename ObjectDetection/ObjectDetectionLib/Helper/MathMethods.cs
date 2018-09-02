using System;
using System.Linq;

namespace ObjectDetectionLib.Helper
{
    public class MathMethods
    {
        public static float Sigmoid(float value)
        {
            var exp = (float)Math.Exp(value);
            return exp / (1.0f + exp);
        }

        public static float[] Softmax(float[] values)
        {
            var maxValue = values.Max();
            var exp = values.Select(v => Math.Exp(v - maxValue));
            var sumExp = exp.Sum();
            return exp.Select(v => (float)(v / sumExp)).ToArray();
        }
    }
}
