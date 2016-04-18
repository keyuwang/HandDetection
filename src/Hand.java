import java.util.*;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.indexer.*;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class Hand {
		
	private static final float SMALLEST_AREA =  600.0f; // ignore smaller contour areas
	
	// HSV ranges defining the hand colour
	private static final int hueLower=0, hueUpper=35, hueShifter=0;
	private static final int hueLower2=150, hueUpper2=255;
	private static final int satLower=10, satUpper=250,satShifter=0;
	private static final int briLower=10, briUpper=240,briShifter=0;
	
	private static final int kernelDist=80;
	
	private static Mat hsvLower,hsvLower2;
	private static Mat hsvUpper,hsvUpper2;
	
	  // defects data for the hand contour
	private ArrayList<Point> fingerTips;	

	
	//flag indicates whether hand is detected
	private static boolean detected;
	private static boolean printed;
	
	// hand details
	private Point cogPt;           // center of gravity (COG) of contour
	private int innerRadius;
	
	private Mat resultImg;
	private Mat hsvImg;
	private Mat imgThreshed,imgThreshed2;
	private Mat kernel,kernel2;
	private MatVector contours;
	private Mat[] list;
	
	/******************Kalman Filter*********************************/
	private KFilter KF;
	
	/*************Static Gesture Recognition******************************/
	private StaticGesture staticGesture;
	private String[] staticGestureName={"None","Ready State","Pressed State","Bloom"};
	
	/*************Dynamic Gesture Recognition******************************/
	private DynamicGesture dynamicGesture;
	private String[] dynamicGestureName={"None","Move","Hold","Click","Bloom"};

	
	public Hand(int height,int width)
	{
		resultImg = new Mat(height,width,CV_8UC4);
		imgThreshed = new Mat(height,width,CV_8UC1);
		imgThreshed2 = new Mat(height,width,CV_8UC1);
		
		kernel=new Mat(8, 8, CV_8U, new Scalar(1d));//opencv erode and dilate kernel
		kernel2=new Mat(kernelDist, kernelDist, CV_8U, new Scalar(1d));
		
		hsvImg = new Mat(height,width,CV_8UC3);
		hsvLower = new Mat(height,width,CV_8UC3,new Scalar(hueLower, satLower, briLower,0));
		hsvUpper = new Mat(height,width,CV_8UC3,new Scalar(hueUpper, satUpper, briUpper,0));
		hsvLower2 = new Mat(height,width,CV_8UC3,new Scalar(hueLower2, satLower, briLower,0));
		hsvUpper2 = new Mat(height,width,CV_8UC3,new Scalar(hueUpper2, satUpper, briUpper,0));
		
		contours = new MatVector();
		list= new Mat[2];
		
		cogPt = new Point();
	    fingerTips = new ArrayList<Point>();
	    
	    KF=new KFilter();
		
	    staticGesture=new StaticGesture();
	    dynamicGesture=new DynamicGesture();
	    
		printed=false;
		
	}
	
	public void update(Mat im)
	{
		if(im.channels()==3)
		{
			cvtColor(im, hsvImg, CV_BGR2HSV);
//			add(hsvImg,new Scalar(hueShifter, satShifter, briShifter,0));
			inRange(hsvImg, hsvLower, hsvUpper, imgThreshed);
			inRange(hsvImg, hsvLower2, hsvUpper2, imgThreshed2);
			add(imgThreshed,imgThreshed2,imgThreshed);
			imgThreshed.copyTo(imgThreshed2);
			cvtColor(im, resultImg, CV_BGR2RGBA);
		}
		else if(im.channels()==1)
		{
			//process depth image
		}
		erode(imgThreshed,imgThreshed,kernel);
		dilate(imgThreshed, imgThreshed, kernel);
		
		innerCircle(imgThreshed2);		
//		cvtColor(imgThreshed2, resultImg, CV_GRAY2RGBA);
		
		list[0]=findBiggestContour(imgThreshed);
		if(list[0]==null)
		{
			detected=false;
			return;
		}
		detected=true;
		
		//extractCog(list[0]);
		
		findFingerTips(list[0]);
		
		KF.update(cogPt);//update Kalman Filter
		
		staticGesture.update(cogPt,fingerTips,innerRadius);
		dynamicGesture.update(staticGesture.getGesture());
		display();//display the result
		
		return;
	}
	
	private Mat findBiggestContour(Mat imgThreshed)
	{
		Mat bigContour = null;
		findContours(imgThreshed, contours, RETR_LIST, CHAIN_APPROX_NONE);
		
		float maxArea = SMALLEST_AREA;
		
		RotatedRect box;
		for(int idx=0;idx<contours.size();idx++)
		{
			box = minAreaRect(contours.get(idx));
			float area = box.size().height()*box.size().width();
			if(area>maxArea)
			{
				maxArea=area;
				bigContour=contours.get(idx);
			}			
		}
				
		
		return bigContour;
	}
	
	private void findFingerTips(Mat approxContour)
	{
	//	Mat approxContour = new Mat();
		Mat hull = new Mat();
		Mat defects = new Mat();
	//	approxPolyDP(bigContour,approxContour,1,true);
		convexHull(approxContour,hull,false,false);
		convexityDefects(approxContour, hull, defects);
		
		IntBufferIndexer hullIdx = hull.createIndexer();
		IntBufferIndexer contourIdx = approxContour.createIndexer();
		
		fingerTips.clear();
		
		int vertex=hullIdx.get(0,hull.rows()-1);
		Point prev=new Point(contourIdx.get(0,vertex,0),contourIdx.get(0,vertex,1));	
		for(int i=0;i<hull.rows();i++)
		{

			vertex=hullIdx.get(0,i);
			Point tip = new Point(contourIdx.get(0,vertex,0),contourIdx.get(0,vertex,1));
			
			if(tip.y()>cogPt.y()+20)continue;//remove point below cogPt
			if(dist(prev,tip)<40)continue;//remove too closed redundant points
			
			int tipAngle=kcurvature(vertex,approxContour,40);
			if(tipAngle>70)continue;//remove big angle

			fingerTips.add(tip);
			
			prev=tip;
		}
						
	    convexHull(approxContour,hull,false,true);
	    
	    list[0]=approxContour;
	    list[1]=hull;
	    
	}
	
	private int angleBetween(Point tip, Point next, Point prev)
	{
		int angle = Math.abs( (int)Math.round( 
                Math.toDegrees(
                      Math.atan2(next.x() - tip.x(), next.y() - tip.y()) -
                      Math.atan2(prev.x() - tip.x(), prev.y() - tip.y())) ));
		if (angle > 180)angle=360-angle;
		return angle;
	}
	
	private int kcurvature(int index,Mat contour,int k)
	{
		
		IntBufferIndexer contourIdx = contour.createIndexer();
		int total=contour.rows();	
		
		int prev=total+index-k;;

		if(prev>=total)
		{
			prev=prev-total;
		}

		int next=index+k;
		if(next>=total)
		{
			next=next-total;
		}
		
    	Point ptStart=new Point(contourIdx.get(0,prev,0),contourIdx.get(0,prev,1));
    	Point ptEnd=new Point(contourIdx.get(0,next,0),contourIdx.get(0,next,1));
    	Point ptFold=new Point(contourIdx.get(0,index,0),contourIdx.get(0,index,1));

//		line(resultImg,ptStart,ptFold,new Scalar(0,255,255,0));
//		line(resultImg,ptEnd,ptFold,new Scalar(0,255,255,0));
		
		return angleBetween(ptFold, ptStart, ptEnd);
	}
	
	private void extractCog(Mat bigContour)
	{
		Moments m = new Moments();
		m=moments(bigContour);
		double m00 = m.m00();
	    double m10 = m.m10();
	    double m01 = m.m01();
	    
	    if (m00 != 0)
	    {
	    	int xCenter = (int) Math.round(m10/m00);
	    	int yCenter = (int) Math.round(m01/m00);
	    	cogPt.x(xCenter);
	    	cogPt.y(yCenter);
	    	circle(resultImg,cogPt,6,new Scalar(0,255,0,0),-1,8,0);
	    }
	}
	
	private void innerCircle(Mat eroded)
	{
		erode(eroded,eroded,kernel2);
		Moments m = new Moments();
		m=moments(eroded,true);
		double m00 = m.m00();
	    double m10 = m.m10();
	    double m01 = m.m01();
	    
	    if (m00 != 0)
	    {
	    	int xCenter = (int) Math.round(m10/m00);
	    	int yCenter = (int) Math.round(m01/m00);
	    	cogPt.x(xCenter);
	    	cogPt.y(yCenter);
	    }
	    int area = countNonZero(eroded);
	    innerRadius=(int) Math.sqrt(area)/4+kernelDist-25;
		return;
	}

	
	private int dist(Point u,Point v)
	{
		return Math.abs(u.x()-v.x())+Math.abs(u.y()-v.y());
	}
	
	public Mat getResult()
	{
		return resultImg; 
	}
	
	public int getFingerNumber()
	{
		return fingerTips.size();
	}
	
	public ArrayList<Point> getFingers()
	{
		return fingerTips;
	}
	
	public boolean isDetected()
	{
		return detected;
	}

/******************debug function*********************************/
	private void display()
	{
		circle(resultImg,KF.getPrediction(),6,new Scalar(255,0,0,0),-1,8,0);
		
		circle(resultImg,cogPt,6,new Scalar(0,255,0,0),-1,8,0);
		circle(resultImg,cogPt,innerRadius,new Scalar(0,0,255,0),1,8,0);
		
		MatVector contourList = new MatVector(list);
	
		RNG rng=new RNG(123456); //openCV Random Number Generator  set seed 123456	
		
		for( int i = 0; i< contourList.size(); i++ )
		{
			Scalar color = new Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255), 0);		
			drawContours(resultImg,
					contourList,
					i, color);
		}
		
		for(int i=0;i<getFingerNumber();i++)
		{
			circle(resultImg,fingerTips.get(i),8,new Scalar(0,0,255,0),-1,8,0);			
			line(resultImg,fingerTips.get(i),cogPt,new Scalar(0,255,255,0),2,8,0);
		}
		
		putText(resultImg, staticGestureName[staticGesture.getGesture()], new Point(0, 20),CV_FONT_HERSHEY_COMPLEX,0.7,new Scalar(0,255,0,0));
		
		putText(resultImg, dynamicGestureName[dynamicGesture.getGesture()], new Point(560, 20),CV_FONT_HERSHEY_COMPLEX,0.7,new Scalar(255,0,0,0));

	}
	
	public void printMat(Mat a)
	{
		if(printed)return;
		printed=true;
		System.out.println("Total= " + a.total());
		System.out.println("Rows= " + a.rows());
		System.out.println("Cols= "+ a.cols());
		System.out.println("Type= " + a.type());
		System.out.println("Channels= " + a.channels());
		
		FloatBufferIndexer fIdx = a.createIndexer();
		for(int y=0;y<a.rows();y++)
		{
			System.out.print("[");
			for(int x=0;x<a.cols();x++)
			{
				System.out.print(fIdx.get(y,x)+" ");
			}
			System.out.print("]\n");
		}
		
	}
}
