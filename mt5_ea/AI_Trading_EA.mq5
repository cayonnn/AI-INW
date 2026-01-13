//+------------------------------------------------------------------+
//|                                            AI_Trading_EA.mq5     |
//|                        AI Execution Layer (ZERO LOGIC)           |
//|                                                                  |
//|  Design Principle (Fund-Grade):                                  |
//|  - Python / AI เป็น Brain                                        |
//|  - MQL5 EA เป็น Dumb Executor                                    |
//|  - ทุกคำสั่ง: Deterministic, Auditable, Fail-safe                |
//|  - No hidden logic                                               |
//+------------------------------------------------------------------+
#property strict
#property version   "2.00"
#property description "AI Execution Layer (ZERO LOGIC)"
#property description "รับคำสั่ง → ตรวจความถูกต้อง → ส่งออเดอร์ → จัดการสถานะ"

#include <Trade/Trade.mqh>

CTrade trade;

//===== INPUT CONFIG =====
input string COMMAND_FILE     = "ai_command.json";   // Command file path
input string RESPONSE_FILE    = "ai_response.json";  // Response file path
input bool   ALLOW_TRADING    = true;                // Allow trading
input int    CHECK_INTERVAL   = 100;                 // Check interval (ms)
input int    MAX_SLIPPAGE     = 30;                  // Max slippage points

//===== TRADE COMMAND STRUCT =====
struct TradeCommand
{
   string action;      // OPEN, CLOSE, MODIFY, CLOSE_ALL
   string symbol;      // Trading symbol
   string direction;   // BUY, SELL
   double volume;      // Lot size
   double sl;          // Stop loss price
   double tp;          // Take profit price
   int    magic;       // Magic number
   string comment;     // Order comment
   ulong  ticket;      // Position ticket (for CLOSE/MODIFY)
};

//===== GLOBAL =====
ulong lastCheck = 0;  // FIX: Changed from datetime to ulong for ms comparison

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("═══════════════════════════════════════════════════════════");
   Print("[EA] AI Execution Layer v2.0 - ZERO LOGIC");
   Print("[EA] Command File: ", COMMAND_FILE);
   Print("[EA] Trading: ", ALLOW_TRADING ? "ENABLED" : "DISABLED");
   Print("═══════════════════════════════════════════════════════════");
   
   trade.SetExpertMagicNumber(900000);
   trade.SetDeviationInPoints(MAX_SLIPPAGE);
   trade.SetTypeFilling(ORDER_FILLING_IOC);
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("[EA] Shutdown. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function - NO LOGIC, ONLY EXECUTE                     |
//+------------------------------------------------------------------+
void OnTick()
{
   if(!ALLOW_TRADING) return;
   
   // Rate limit
   ulong now = GetTickCount64();
   if(now - lastCheck < (ulong)CHECK_INTERVAL) return;
   lastCheck = now;
   
   // Check for command file
   if(!FileIsExist(COMMAND_FILE, FILE_COMMON))
      return;
   
   // Load command
   TradeCommand cmd;
   if(!LoadCommand(cmd))
   {
      WriteResponse("ERROR", "Failed to load command");
      FileDelete(COMMAND_FILE, FILE_COMMON);
      return;
   }
   
   // Validate command
   if(!ValidateCommand(cmd))
   {
      WriteResponse("ERROR", "Invalid command");
      FileDelete(COMMAND_FILE, FILE_COMMON);
      return;
   }
   
   // Execute command (NO LOGIC - JUST EXECUTE)
   bool success = ExecuteCommand(cmd);
   
   // Write response
   WriteResponse(success ? "OK" : "FAILED", trade.ResultComment());
   
   // Clear command file (CRITICAL)
   FileDelete(COMMAND_FILE, FILE_COMMON);
}

//+------------------------------------------------------------------+
//| Load Command from JSON file - NO INTERPRETATION                   |
//+------------------------------------------------------------------+
bool LoadCommand(TradeCommand &cmd)
{
   int handle = FileOpen(COMMAND_FILE, FILE_READ | FILE_ANSI | FILE_COMMON);
   if(handle == INVALID_HANDLE)
   {
      Print("[EA] Cannot open command file");
      return false;
   }
   
   // Read entire file at once
   string json = FileReadString(handle, (int)FileSize(handle));
   FileClose(handle);
   
   Print("[EA] Raw JSON: ", json);
   
   if(StringLen(json) < 10)
   {
      Print("[EA] Empty or invalid JSON, len=", StringLen(json));
      return false;
   }
   
   // Parse JSON (simple extraction)
   cmd.action    = ExtractString(json, "action");
   cmd.symbol    = ExtractString(json, "symbol");
   cmd.direction = ExtractString(json, "direction");
   cmd.volume    = ExtractDouble(json, "volume");
   cmd.sl        = ExtractDouble(json, "sl");
   cmd.tp        = ExtractDouble(json, "tp");
   cmd.magic     = (int)ExtractDouble(json, "magic");
   cmd.comment   = ExtractString(json, "comment");
   cmd.ticket    = (ulong)ExtractDouble(json, "ticket");
   
   Print("[EA] Command loaded: ", cmd.action, " ", cmd.symbol, " ", cmd.direction);
   
   return true;
}

//+------------------------------------------------------------------+
//| Validate Command - SAFETY ONLY, NO LOGIC                          |
//+------------------------------------------------------------------+
bool ValidateCommand(TradeCommand &cmd)
{
   // Debug: show what was parsed
   Print("[EA] Parsed values:");
   Print("[EA]   action='", cmd.action, "'");
   Print("[EA]   symbol='", cmd.symbol, "'");
   Print("[EA]   direction='", cmd.direction, "'");
   Print("[EA]   volume=", cmd.volume);
   
   // Action must exist
   if(cmd.action == "")
   {
      Print("[EA] Missing action");
      return false;
   }
   
   // For OPEN action
   if(cmd.action == "OPEN")
   {
      if(cmd.symbol == "" || cmd.direction == "" || cmd.volume <= 0)
      {
         Print("[EA] Invalid OPEN parameters: symbol=", cmd.symbol, " dir=", cmd.direction, " vol=", cmd.volume);
         return false;
      }
      
      // Check symbol exists
      if(!SymbolSelect(cmd.symbol, true))
      {
         Print("[EA] Symbol not found: ", cmd.symbol);
         return false;
      }
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Execute Command - PURE EXECUTION, ZERO BRAIN                      |
//+------------------------------------------------------------------+
bool ExecuteCommand(TradeCommand &cmd)
{
   trade.SetExpertMagicNumber(cmd.magic > 0 ? cmd.magic : 900000);
   
   //--- OPEN ---
   if(cmd.action == "OPEN")
   {
      return ExecuteOpen(cmd);
   }
   
   //--- CLOSE ---
   if(cmd.action == "CLOSE")
   {
      return ExecuteClose(cmd);
   }
   
   //--- CLOSE_ALL ---
   if(cmd.action == "CLOSE_ALL")
   {
      return ExecuteCloseAll(cmd);
   }
   
   //--- MODIFY ---
   if(cmd.action == "MODIFY")
   {
      return ExecuteModify(cmd);
   }
   
   Print("[EA] Unknown action: ", cmd.action);
   return false;
}

//+------------------------------------------------------------------+
//| Execute OPEN - No decision, just send order                       |
//+------------------------------------------------------------------+
bool ExecuteOpen(TradeCommand &cmd)
{
   bool result = false;
   
   if(cmd.direction == "BUY")
   {
      result = trade.Buy(cmd.volume, cmd.symbol, 0, cmd.sl, cmd.tp, cmd.comment);
   }
   else if(cmd.direction == "SELL")
   {
      result = trade.Sell(cmd.volume, cmd.symbol, 0, cmd.sl, cmd.tp, cmd.comment);
   }
   
   if(result)
      Print("[EA] Order opened: ", trade.ResultOrder());
   else
      Print("[EA] Order failed: ", trade.ResultComment());
   
   return result;
}

//+------------------------------------------------------------------+
//| Execute CLOSE - Close by ticket or magic                          |
//+------------------------------------------------------------------+
bool ExecuteClose(TradeCommand &cmd)
{
   if(cmd.ticket > 0)
   {
      return trade.PositionClose(cmd.ticket);
   }
   
   // Close by magic
   return CloseByMagic(cmd.magic);
}

//+------------------------------------------------------------------+
//| Execute CLOSE_ALL - Close all by magic                            |
//+------------------------------------------------------------------+
bool ExecuteCloseAll(TradeCommand &cmd)
{
   int magic = cmd.magic > 0 ? cmd.magic : 900000;
   return CloseByMagic(magic);
}

//+------------------------------------------------------------------+
//| Execute MODIFY - Modify SL/TP                                     |
//+------------------------------------------------------------------+
bool ExecuteModify(TradeCommand &cmd)
{
   if(cmd.ticket == 0)
   {
      Print("[EA] No ticket for MODIFY");
      return false;
   }
   
   return trade.PositionModify(cmd.ticket, cmd.sl, cmd.tp);
}

//+------------------------------------------------------------------+
//| Close positions by magic number                                   |
//+------------------------------------------------------------------+
bool CloseByMagic(int magic)
{
   int closed = 0;
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
      {
         if(PositionGetInteger(POSITION_MAGIC) == magic)
         {
            if(trade.PositionClose(ticket))
               closed++;
         }
      }
   }
   
   Print("[EA] Closed ", closed, " positions");
   return closed > 0;
}

//+------------------------------------------------------------------+
//| Write response to file (ATOMIC - prevents race condition)         |
//+------------------------------------------------------------------+
void WriteResponse(string status, string message)
{
   string response = StringFormat(
      "{\"status\":\"%s\",\"message\":\"%s\",\"time\":\"%s\"}",
      status, message, TimeToString(TimeCurrent())
   );
   
   Print("[EA] Writing response: ", response);
   
   // Step 1: Write to temp file
   string tmp_file = "ai_response.tmp";
   int handle = FileOpen(tmp_file, FILE_WRITE | FILE_COMMON | FILE_ANSI);
   if(handle == INVALID_HANDLE) 
   {
      Print("[EA] Failed to open temp file!");
      return;
   }
   
   FileWriteString(handle, response);
   FileClose(handle);
   
   // Step 2: Atomic move to final file
   if(FileIsExist(RESPONSE_FILE, FILE_COMMON))
      FileDelete(RESPONSE_FILE, FILE_COMMON);
   
   if(FileMove(tmp_file, 0, RESPONSE_FILE, FILE_COMMON | FILE_REWRITE))
   {
      Print("[EA] Response written successfully (atomic)");
   }
   else
   {
      Print("[EA] FileMove failed, fallback to direct write");
      // Fallback: direct write if FileMove fails
      int h2 = FileOpen(RESPONSE_FILE, FILE_WRITE | FILE_COMMON | FILE_ANSI);
      if(h2 != INVALID_HANDLE)
      {
         FileWriteString(h2, response);
         FileClose(h2);
      }
   }
}

//+------------------------------------------------------------------+
//| Extract string value from JSON (robust version)                   |
//+------------------------------------------------------------------+
string ExtractString(string json, string key)
{
   // Build pattern: "key":
   string pattern = "";
   pattern += CharToString(34);  // " character
   pattern += key;
   pattern += CharToString(34);  // " character
   pattern += ":";
   
   int pos = StringFind(json, pattern);
   if(pos < 0) 
   {
      Print("[EA] Key not found: ", key, " in: ", StringSubstr(json, 0, 100));
      return "";
   }
   
   int start = pos + StringLen(pattern);
   
   // Skip whitespace
   while(start < StringLen(json) && StringGetCharacter(json, start) == ' ')
      start++;
   
   // Check for quoted string (char code 34 = ")
   if(StringGetCharacter(json, start) == 34)
   {
      start++;
      int end = start;
      while(end < StringLen(json) && StringGetCharacter(json, end) != 34)
         end++;
      string result = StringSubstr(json, start, end - start);
      return result;
   }
   
   // Unquoted value (number, null, bool)
   int end = start;
   while(end < StringLen(json) && 
         StringGetCharacter(json, end) != ',' && 
         StringGetCharacter(json, end) != '}')
      end++;
   
   return StringSubstr(json, start, end - start);
}

//+------------------------------------------------------------------+
//| Extract double value from JSON                                    |
//+------------------------------------------------------------------+
double ExtractDouble(string json, string key)
{
   string val = ExtractString(json, key);
   if(val == "") return 0;
   return StringToDouble(val);
}

//+------------------------------------------------------------------+
//| END OF FILE - ZERO LOGIC EA                                       |
//+------------------------------------------------------------------+
