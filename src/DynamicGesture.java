
public class DynamicGesture {
	
	public static final int CYCLE=40;
	
	public static final int NONE=0;
	public static final int MOVE=1;
	public static final int HOLD=2;
	public static final int CLICK=3;
	public static final int BLOOM=4;
	private int down=0,up=0,move=0,hold=0;
	private int state=0;
	private int preStaticState=0;

	public DynamicGesture() {
		// TODO Auto-generated constructor stub
	}
	
	public void update(int staticState)
	{
		if(down>0)down--;
		if(up>0)up--;
		state=NONE;
		if(staticState==1)
		{
			if(preStaticState==1)state=MOVE;
			if(preStaticState==2)up=CYCLE;
		}
		
		if(staticState==2)
		{
			if(preStaticState==2)state=HOLD;
			if(preStaticState==1)down=CYCLE;
		}
		if(staticState==3&&preStaticState==NONE)state=BLOOM;
		
		if(down>0&&up>down)
		{
			state=CLICK;
		}
		
		preStaticState=staticState;
	}
	
	public int getGesture()
	{
		return state;
	}
}
