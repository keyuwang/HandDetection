
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.FloatBufferIndexer;
import org.bytedeco.javacpp.opencv_video.KalmanFilter;

import static org.bytedeco.javacpp.opencv_core.*;

public class KFilter {
	
	private KalmanFilter KF;
	private float[] f ={1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1};
	private FloatPointer p;
	private Mat F,H,Q,R,P;//kalman parameter matrix
	private Mat measurement,estimated,prediction;
	
	public KFilter()
	{
		KF = new KalmanFilter(4,2,0,CV_32F);
		p= new FloatPointer(f);
		F=new Mat(4,4,CV_32F,p);
		KF.transitionMatrix(F);
		
		H=new Mat(2, 4,CV_32F);
		setIdentity(H);
		KF.measurementMatrix(H);
		
		Q=new Mat(4, 4,CV_32F);
		setIdentity(Q,Scalar.all(1e-4));
		KF.processNoiseCov(Q);
		
		R=new Mat(2, 2,CV_32F);
		setIdentity(R,Scalar.all(1e-2));
		KF.measurementNoiseCov(R);
		
		P=new Mat(4, 4,CV_32F);
		setIdentity(P,Scalar.all(1));
		KF.errorCovPost(P);
		
		
		measurement=new Mat(2,1,CV_32F);
		estimated=new Mat(4,1,CV_32F);	
		prediction=new Mat(4,1,CV_32F);	
		
	}
	
	public void update(Point input)
	{
		
		FloatBufferIndexer pIdx=measurement.createIndexer();		
		pIdx.put(0, input.x());
		pIdx.put(1, input.y());
		
		estimated=KF.correct(measurement);
		prediction=KF.predict();
	}
	
	public Point getEstimated()
	{
		FloatBufferIndexer pIdx = estimated.createIndexer();
		Point output=new Point((int)pIdx.get(0),(int)pIdx.get(1));
		return output;
	}
	
	public Point getPrediction()
	{
		FloatBufferIndexer pIdx = prediction.createIndexer();
		Point output=new Point((int)pIdx.get(0),(int)pIdx.get(1));
		return output;
	}

	
}
