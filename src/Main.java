import org.bytedeco.javacv.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_calib3d.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;

public class Main {
    public static void main(String[] args) throws Exception {

        FrameGrabber grabber = FrameGrabber.createDefault(1);//replace 0 for default camera
        grabber.start();

        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        Mat  grabbedImage = converter.convert(grabber.grab());

        int width  = grabbedImage.rows();
        int height = grabbedImage.cols();
        Mat grayImage  = new Mat(width, height, CV_8UC1); 
       
        Hand hand=new Hand(width, height);

        CanvasFrame frame = new CanvasFrame("Some Title", CanvasFrame.getDefaultGamma()/grabber.getGamma());


        while (frame.isVisible() && (grabbedImage = converter.convert(grabber.grab())) != null) {
        	
        	hand.update(grabbedImage);
            Frame outFrame = converter.convert(hand.getResult());
            frame.showImage(outFrame);       
        }        
        //  for(;;);
        frame.dispose();
        grabber.stop(); 
    }
    
}