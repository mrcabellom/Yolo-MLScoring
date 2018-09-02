using ObjectDetectionLib.Services;
using ObjectDetectionWeb.ViewModels;
using System.Web.Http;


namespace ObjectDetectionWeb.Controllers
{
    [RoutePrefix("api/predictive")]
    public class PredictiveController : ApiController
    {
        [HttpPost]
        public IHttpActionResult LocalPrediction([FromBody] ImageViewModel image)
        {
            var result = YoloPrediction.ObtainBoundingBoxes(image.EncodedImage);
            return Ok(result);
        }        
    }
}