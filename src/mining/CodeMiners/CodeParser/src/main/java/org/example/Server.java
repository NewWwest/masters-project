package org.example;

import java.io.IOException;
import java.io.OutputStream;
import java.io.UnsupportedEncodingException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URLDecoder;
import java.util.LinkedHashMap;
import java.util.Map;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

public class Server {
    public static void main(String[] args) throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress(8000), 0);
        server.createContext("/parse", new MyHandler());
        server.setExecutor(null); // creates a default executor
        server.start();
    }

    static class MyHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange t) throws IOException {
            OutputStream os = t.getResponseBody();
            try{
                URI uri = t.getRequestURI();
                String query = uri.getQuery();
                Map<String, String> paramsx = splitQuery(query);
                String file_new = paramsx.get("file_new");
                String file_old = paramsx.get("file_new");
                String out_ast = paramsx.get("out_ast");
                String out_action = paramsx.get("out_action");
                GumTreeDiffParserProxy.ProcessFiles(file_new, file_old, out_ast, out_action);
                
                String response = "ok";
                t.sendResponseHeaders(200, response.length());
                os.write(response.getBytes());
            }
            catch(Exception e) {
                String response = "nok";
                t.sendResponseHeaders(500, response.length());
                os.write(response.getBytes());
            }
            finally{
                os.close();
            }
        }
    }
    
    public static Map<String, String> splitQuery(String query) throws UnsupportedEncodingException {
        Map<String, String> query_pairs = new LinkedHashMap<String, String>();
        String[] pairs = query.split("&");
        for (String pair : pairs) {
            int idx = pair.indexOf("=");
            query_pairs.put(URLDecoder.decode(pair.substring(0, idx), "UTF-8"), URLDecoder.decode(pair.substring(idx + 1), "UTF-8"));
        }
        return query_pairs;
    }
}
