import java.util.ArrayList;

import org.bytedeco.javacpp.opencv_core.*;


public class StaticGesture {
	public static final int NONE=0;
	public static final int POINTING=1;
	public static final int PINCH=2;
	public static final int OTHERS=3;
	private int state=0;
	
	public StaticGesture()
	{
		
	}
	
	public void update(Point cog, ArrayList<Point> fingerTips)
	{
		int fingerNumber=fingerTips.size();
		switch (fingerNumber)
		{
		case 0: state=NONE; break;
		case 1: state=POINTING; break;
		case 2: state=PINCH; break;		
		default: state=OTHERS; break;
		}
	}
	
	public int getGesture()
	{
		return state;
	}

}
