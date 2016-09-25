import java.util.ArrayList;

import org.bytedeco.javacpp.opencv_core.*;


public class StaticGesture {
	public static final int NONE=0;
	public static final int READY=1;
	public static final int PRESSED=2;
	public static final int ZOOM=3;
	public static final int BLOOM=4;
	private int state=0;
	private int indexFingerMax=0;

	private int indexFingerLen=0;
	
	private int fingerNumber=0;
	
	private int zoomDist=0;
	
	private float ratio=1.0f;
	
	private Point tip;
	
	private int[] cache;
	
	public StaticGesture()
	{
		tip=new Point();
		cache = new int[8];
		
	}
	
	public void update(Point cog, ArrayList<Point> fingerTips,int radius)
	{
		fingerNumber=fingerTips.size();
		if(fingerNumber==5)
		{
			for(int i=0;i<5;i++)
			{
				int tmpdist=dist(fingerTips.get(i),cog);
				if(tmpdist>indexFingerMax)indexFingerMax=tmpdist;
			}
		}
		if(fingerNumber==2)
		{
			fingerNumber=6;
			zoomDist=dist(fingerTips.get(0),fingerTips.get(1));
		}
			
		if(fingerNumber==1)
		{
			indexFingerLen=dist(fingerTips.get(0),cog);
//			if(indexFingerLen>indexFingerMax)indexFingerMax=indexFingerLen;
			ratio=0.9f*indexFingerMax/indexFingerLen;
			getTip(fingerTips.get(0),cog);
			int dif=indexFingerMax-radius;
			int threshold=radius+dif*4/5;
			if(indexFingerLen<threshold)fingerNumber=2;
		}
		
		switch (fingerNumber)
		{
		case 1: state=READY; break;
		case 2: state=PRESSED; break;
		case 6: state=ZOOM; break;
		case 5: state=BLOOM; break;	
		default: state=NONE; break;
		}
		
		push(state);
		
	}
	
	private int dist(Point u,Point v)
	{
		return (int) Math.sqrt(
				((u.x()-v.x())*(u.x()-v.x())
				+(u.y()-v.y())*(u.y()-v.y())));
	}
	
	private void getTip(Point finger,Point cog)
	{
		tip.x((int) (ratio*(finger.x()-cog.x())+cog.x()));
		tip.y((int) (ratio*(finger.y()-cog.y())+cog.y()));
	}
	
	private void push(int x)
	{
		for(int i=1;i<cache.length;i++)
		{
			cache[i-1]=cache[i];
		}
		cache[cache.length-1]=x;
	}
	
	public int getGesture()
	{
		if(state!=READY && state!=PRESSED)
		{
			
		}
		return state;
	}
	
	public Point getTipPostion()
	{
		if(state!=READY&&state!=PRESSED)return null;
		return tip;
	}
	
	public int getZoomDist()
	{
		return zoomDist;
	}
	
	

}
