import java.util.ArrayList;

import org.bytedeco.javacpp.opencv_core.*;


public class StaticGesture {
	public static final int NONE=0;
	public static final int READY=1;
	public static final int PRESSED=2;
	public static final int BLOOM=3;
	private int state=0;
	private int indexFingerMax=0;
	private int indexFingerMin=1000;
	public StaticGesture()
	{
		
	}
	
	public void update(Point cog, ArrayList<Point> fingerTips,int radius)
	{
		int fingerNumber=fingerTips.size();
		if(fingerNumber==2)fingerNumber=3;
		if(fingerNumber==1)
		{
			int indexFingerLen=dist(fingerTips.get(0),cog);
			if(indexFingerLen<indexFingerMin)indexFingerMin=indexFingerLen;
			if(indexFingerLen>indexFingerMax)indexFingerMax=indexFingerLen;
			int dif=indexFingerMax-radius;
			int threshold=radius+dif*2/3;
			if(indexFingerLen<threshold)fingerNumber=2;
		}
		
		switch (fingerNumber)
		{
		case 1: state=READY; break;
		case 2: state=PRESSED; break;
		case 5: state=BLOOM; break;	
		default: state=NONE; break;
		}
	}
	
	private int dist(Point u,Point v)
	{
		return Math.abs(u.x()-v.x())+Math.abs(u.y()-v.y());
	}

	
	public int getGesture()
	{
		return state;
	}

}
