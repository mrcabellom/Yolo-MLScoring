using Microsoft.ML.Scoring;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;

namespace YoloInference
{
    public partial class YoloModel
    {
        private static readonly object padlock = new object();
        const string modelName = "YoloModel";
        private ModelManager manager;

        private static List<string> evaluateInputNames = new List<string> { "input0" };
        private static List<string> evaluateOutputNames = new List<string> { "output0" };
        private static readonly Lazy<YoloModel> lazy = new Lazy<YoloModel>(() => new YoloModel());
        public static YoloModel Instance { get { return lazy.Value; } }

        /// <summary>
        /// Returns an instance of YoloModel model.
        /// </summary>
        private YoloModel()
        {
            string codeBase = Assembly.GetExecutingAssembly().CodeBase;
            UriBuilder uri = new UriBuilder(codeBase);
            string dllpath = Uri.UnescapeDataString(uri.Path);
            string modelpath = Path.Combine(Path.GetDirectoryName(dllpath), "YoloModel");
            string path = Path.Combine(modelpath, "00000001");
            manager = new ModelManager(path, true);
            manager.InitModel(modelName, int.MaxValue);
        }

        /// <summary>
        /// Runs inference on YoloModel model for a batch of inputs.
        /// The shape of each input is the same as that for the non-batch case above.
        /// </summary>
        public IEnumerable<IEnumerable<float>> Evaluate(IEnumerable<IEnumerable<float>> input0Batch)
        {

            List<float> input0Combined = new List<float>();
            foreach (var input in input0Batch)
            {
                input0Combined.AddRange(input);
            }

            List<Tensor> result = manager.RunModel(
                modelName,
                int.MaxValue,
                evaluateInputNames,
                new List<Tensor> { new Tensor(input0Combined, new List<long> { input0Batch.LongCount(), 3, 416, 416 }) },
                evaluateOutputNames
            );

            int output0BatchNum = (int)result[0].GetShape()[0];
            int output0BatchSize = (int)result[0].GetShape().Aggregate((a, x) => a * x) / output0BatchNum;
            for (int batchNum = 0, offset = 0; batchNum < output0BatchNum; batchNum++, offset += output0BatchSize)
            {
                List<float> tmp = new List<float>();
                result[0].CopyTo(tmp, offset, output0BatchSize);
                yield return tmp;
            }
        }
    } // END OF CLASS
} // END OF NAMESPACE
